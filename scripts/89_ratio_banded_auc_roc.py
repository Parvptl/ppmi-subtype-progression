"""ROC and AUC for the ratio-banded CT-HMM against its baseline, same cases.

Reads the held-out probabilities the ratio-banded experiment already produced.
Nothing is refitted, no model is touched, and the frozen run directory is not
written to.

AUC answers a question the accuracy contrast cannot. Accuracy depends on the
argmax decision rule, and the ratio-banded model moved its decision boundary
toward PIGD -- 890 PIGD calls against 800 -- so a fall in accuracy is
consistent either with worse *ranking* or with the same ranking read at a worse
threshold. AUC is threshold-free and calibration-invariant: it measures only
how well each arm ranks a PIGD next visit above a TD one. If the AUCs coincide
while accuracy differs, the loss is threshold placement rather than information.

Conventions follow scripts/66_roc_auc_comparison.py so the numbers here are
comparable with the project's existing AUC analysis: PIGD is the positive
class, the score is the conditional two-class P_PIGD, the evaluation set is
eligible & two_class_defined with Indeterminate targets excluded, and every
interval resamples *patients* rather than predictions.

    python scripts/89_ratio_banded_auc_roc.py --out <dir>
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.clinical_profile import auc as manual_auc  # noqa: E402
from ppmi_subtype.plots import (  # noqa: E402
    GRID, INK_MUTED, SURFACE, _finish, _style,
)

RUN = "experiments/20261104T152154Z_cthmm_ratio_banded_observations"
BASELINE = "CT-HMM (latent, interval-aware)"
BANDED = "CT-HMM (subtype x ratio band)"
CONTROL = "CT-HMM (subtype labels, joint code path)"

ACCENT = "#1f4e79"
WARM = "#eb6834"
NEUTRAL = "#b9b8b4"

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


def _by_patient(patno, columns):
    """Group row tuples by patient, preserving order."""
    out: dict = {}
    for key, row in zip(patno, zip(*columns)):
        out.setdefault(key, []).append(row)
    return out, np.array(list(out.keys()), dtype=object)


def patient_bootstrap(score, positive, patno, metric, seed=42, n_boot=2000):
    """Percentile CI for a ranking metric, resampling patients with replacement."""
    score = np.asarray(score, float)
    positive = np.asarray(positive).astype(bool)
    by_patient, patients = _by_patient(np.asarray(patno), (score, positive))

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for i in pick for r in by_patient[patients[i]]]
        s = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        if y.all() or not y.any():
            continue
        draws[b] = metric(y, s)
    draws = draws[np.isfinite(draws)]
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
            int(len(draws)))


def patient_bootstrap_diff(sa, sb, positive, patno, metric, seed=42, n_boot=2000):
    """Percentile CI for metric(a) - metric(b), paired, resampling patients.

    Both arms are recomputed on the *same* resampled patients, so the interval
    reflects the within-patient correlation between the two sets of scores.
    """
    sa, sb = np.asarray(sa, float), np.asarray(sb, float)
    positive = np.asarray(positive).astype(bool)
    by_patient, patients = _by_patient(np.asarray(patno), (sa, sb, positive))

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for k in pick for r in by_patient[patients[k]]]
        a = np.array([r[0] for r in rows])
        b_ = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows])
        if y.all() or not y.any():
            continue
        draws[i] = metric(y, a) - metric(y, b_)
    kept = draws[np.isfinite(draws)]
    return {"delta": float(metric(positive, sa) - metric(positive, sb)),
            "ci_low": float(np.quantile(kept, 0.025)),
            "ci_high": float(np.quantile(kept, 0.975)),
            "n_bootstrap": int(len(kept))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ------------------------------------------------------- evaluation set
    section("1. THE EVALUATION SET")
    src = ROOT / RUN / "next_visit_predictions.csv"
    say(f"  source   {RUN}/next_visit_predictions.csv")
    say(f"  sha256   {sha(src)}")
    pred = pd.read_csv(src, low_memory=False)

    key = ["PATNO", "current_event", "target_event"]
    arms: dict[str, pd.DataFrame] = {}
    for name in (BASELINE, BANDED, CONTROL):
        d = pred[pred["model"] == name].copy()
        d = d[d["eligible"] & d["two_class_defined"]]
        arms[name] = d.sort_values(key).reset_index(drop=True)
        say(f"  {name:<44} {len(d):,} scored")

    ref = arms[BASELINE]
    for name, d in arms.items():
        if not (d[key].to_numpy() == ref[key].to_numpy()).all():
            raise SystemExit(f"row alignment differs for {name}")
        if not (d["actual_next_label"].to_numpy()
                == ref["actual_next_label"].to_numpy()).all():
            raise SystemExit(f"the arms disagree on the actual label for {name}")
    say("  every arm scores the identical rows, in the same order, against the")
    say("  same actual labels -- asserted on all three, not assumed.")

    y = (ref["actual_next_label"] == "PIGD").to_numpy()
    patno = ref["PATNO"].to_numpy()
    n_pat = int(len(np.unique(patno)))
    say(f"\n  cases                {len(y):,}")
    say(f"  patients             {n_pat:,}")
    say(f"  actual PIGD          {int(y.sum()):,}  ({y.mean():.2%})")
    say(f"  actual TD            {int((~y).sum()):,}")
    say("  Indeterminate targets are excluded, as everywhere else in the project.")
    say("  PIGD is the positive class and the score is the conditional two-class")
    say("  P_PIGD; because P_TD = 1 - P_PIGD, taking TD as positive gives the")
    say("  identical AUC, so one number covers both directions.")

    # Accuracy is recomputed here so there is proof the AUC below is measured on
    # exactly the set the published accuracy was measured on.
    section("2. THE SET MATCHES THE PUBLISHED ACCURACY")
    published = pd.read_csv(ROOT / RUN / "model_comparison.csv")
    acc_rows = []
    for name, d in arms.items():
        acc = float((d["predicted_state"] == d["actual_next_label"]).mean())
        want = float(published.loc[published["model"] == name, "accuracy"].iloc[0])
        agree = bool(np.isclose(acc, want, atol=5e-5))
        acc_rows.append({"model": name, "accuracy_recomputed": round(acc, 6),
                         "accuracy_published": want, "agrees": agree})
        say(f"  {name:<44} {acc:.6f} vs {want:.4f} published   "
            f"{'OK' if agree else 'MISMATCH'}")
        if not agree:
            raise SystemExit(f"accuracy mismatch for {name}")
    pd.DataFrame(acc_rows).to_csv(out / "accuracy_crosscheck.csv", index=False)

    # ---------------------------------------------------------------- AUC
    section("3. AUC-ROC")
    say("  Intervals resample the 899 patients, not the 3,029 predictions: a")
    say("  patient contributing fourteen correlated rows would otherwise make")
    say(f"  the interval too narrow. {args.n_boot:,} resamples, seed 42.")
    say("")
    labels = {BASELINE: "baseline (subtype only)",
              BANDED: "ratio-banded (subtype x band)",
              CONTROL: "control (baseline via joint path)"}
    rows, curves = [], {}
    for name, d in arms.items():
        score = d["P_PIGD"].to_numpy(float)
        a = roc_auc_score(y, score)
        a_manual = manual_auc(score, y)
        if not np.isclose(a, a_manual, atol=1e-9):
            raise SystemExit(f"AUC disagreement for {name}: {a} vs {a_manual}")
        lo, hi, nd = patient_bootstrap(score, y, patno, roc_auc_score,
                                       n_boot=args.n_boot)
        ap_ = average_precision_score(y, score)
        ap_lo, ap_hi, _ = patient_bootstrap(score, y, patno,
                                            average_precision_score,
                                            n_boot=args.n_boot)
        fpr, tpr, _ = roc_curve(y, score)
        curves[name] = (fpr, tpr)
        rows.append({"model": name, "arm": labels[name], "n": len(y),
                     "n_patients": n_pat, "n_positive": int(y.sum()),
                     "auc_roc": a, "auc_ci_low": lo, "auc_ci_high": hi,
                     "average_precision": ap_, "ap_ci_low": ap_lo,
                     "ap_ci_high": ap_hi, "n_bootstrap": nd,
                     "n_distinct_scores": int(len(np.unique(score)))})
        say(f"  {labels[name]:<34} AUC {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        say(f"  {'':<34} PR-AUC {ap_:.4f}  95% CI [{ap_lo:.4f}, {ap_hi:.4f}]")
    say("")
    say("  AUC agrees with the project's own rank-based implementation to 1e-9")
    say("  for every arm. PR-AUC is secondary and reported because PIGD is the")
    say("  minority class at 30.2%; it is not the pre-specified metric.")
    table = pd.DataFrame(rows)
    table.round(6).to_csv(out / "auc_metrics.csv", index=False)

    # ----------------------------------------------------------- contrasts
    section("4. PAIRED CONTRASTS")
    contrasts = []
    for a_name, b_name, what in (
        (BANDED, BASELINE, "ratio bands added to the subtype label"),
        (CONTROL, BASELINE, "control: the baseline model through the joint path"),
    ):
        sa = arms[a_name]["P_PIGD"].to_numpy(float)
        sb = arms[b_name]["P_PIGD"].to_numpy(float)
        d_auc = patient_bootstrap_diff(sa, sb, y, patno, roc_auc_score,
                                       n_boot=args.n_boot)
        d_ap = patient_bootstrap_diff(sa, sb, y, patno, average_precision_score,
                                      n_boot=args.n_boot)
        verdict = ("includes zero" if d_auc["ci_low"] <= 0 <= d_auc["ci_high"]
                   else "excludes zero")
        contrasts.append({"contrast": what, "model_a": a_name, "model_b": b_name,
                          "auc_a": roc_auc_score(y, sa),
                          "auc_b": roc_auc_score(y, sb),
                          "auc_delta": d_auc["delta"],
                          "auc_delta_ci_low": d_auc["ci_low"],
                          "auc_delta_ci_high": d_auc["ci_high"],
                          "ap_delta": d_ap["delta"],
                          "ap_delta_ci_low": d_ap["ci_low"],
                          "ap_delta_ci_high": d_ap["ci_high"],
                          "n_paired": len(y), "n_patients": n_pat,
                          "n_bootstrap": d_auc["n_bootstrap"],
                          "ci_includes_zero": verdict == "includes zero"})
        say(f"  {what}")
        say(f"    AUC    {d_auc['delta']:+.4f}  95% CI "
            f"[{d_auc['ci_low']:+.4f}, {d_auc['ci_high']:+.4f}]  -> {verdict}")
        say(f"    PR-AUC {d_ap['delta']:+.4f}  95% CI "
            f"[{d_ap['ci_low']:+.4f}, {d_ap['ci_high']:+.4f}]")
    pd.DataFrame(contrasts).round(6).to_csv(out / "auc_contrasts.csv", index=False)
    say("")
    say("  Both arms are recomputed on the same resampled patients, so these are")
    say("  paired intervals and not the difference of the two marginal ones.")

    # ------------------------------------- where the decision rule operates
    section("5. WHERE THE DECISION RULE OPERATES")
    say("  AUC integrates over every threshold, but the reported accuracy uses")
    say("  one: argmax, which calls PIGD when the conditional P_PIGD exceeds")
    say("  0.5. If the arms differ in AUC but not at that single point, the")
    say("  ranking gain sits in a region of the curve the decision rule never")
    say("  visits, and accuracy cannot see it.")
    say("")
    op_rows = []
    for name, d in arms.items():
        score = d["P_PIGD"].to_numpy(float)
        called = (d["predicted_state"].to_numpy() == "PIGD")
        tpr_op = float(called[y].mean())
        fpr_op = float(called[~y].mean())
        # McClish-standardised partial AUC: 0.5 is chance within the region.
        p20 = roc_auc_score(y, score, max_fpr=0.2)
        op_rows.append({"model": name, "arm": labels[name],
                        "n_called_PIGD": int(called.sum()),
                        "operating_tpr": tpr_op, "operating_fpr": fpr_op,
                        "partial_auc_fpr_le_0.2_mcclish": float(p20)})
        say(f"  {labels[name]:<34} operates at FPR {fpr_op:.4f}, TPR {tpr_op:.4f}"
            f"   ({int(called.sum()):,} PIGD calls)")
        say(f"  {'':<34} partial AUC, FPR <= 0.2: {p20:.4f}")
    pd.DataFrame(op_rows).round(6).to_csv(out / "operating_points.csv", index=False)

    dp = patient_bootstrap_diff(
        arms[BANDED]["P_PIGD"].to_numpy(float),
        arms[BASELINE]["P_PIGD"].to_numpy(float), y, patno,
        lambda yy, ss: roc_auc_score(yy, ss, max_fpr=0.2), n_boot=args.n_boot)
    pverdict = ("includes zero" if dp["ci_low"] <= 0 <= dp["ci_high"]
                else "excludes zero")
    say("")
    say(f"  banded - baseline, partial AUC (FPR <= 0.2): {dp['delta']:+.4f}"
        f"  95% CI [{dp['ci_low']:+.4f}, {dp['ci_high']:+.4f}]  -> {pverdict}")
    pd.DataFrame([{"contrast": "banded - baseline, partial AUC FPR<=0.2",
                   **dp}]).round(6).to_csv(
        out / "partial_auc_contrast.csv", index=False)

    # ---------------------------------------------------------- breakdowns
    section("6. AUC BY STRATUM")
    say("  Within a stratum the ranking problem is harder than the pooled one,")
    say("  because the strongest predictor is held closer to constant. These")
    say("  are not comparable to the pooled AUC above, only to each other.")
    strata = []
    for field, name in (("current_label", "current label"),
                        ("n_history", "prior visits")):
        col = ref[field].to_numpy()
        for level in (["TD", "Indeterminate", "PIGD"] if field == "current_label"
                      else sorted(np.unique(col))):
            m = (col == level)
            row = {"stratum": name, "level": str(level), "n": int(m.sum()),
                   "n_positive": int(y[m].sum())}
            for arm_name, short in ((BASELINE, "baseline"), (BANDED, "banded")):
                s = arms[arm_name]["P_PIGD"].to_numpy(float)[m]
                row[f"auc_{short}"] = (
                    float(roc_auc_score(y[m], s))
                    if m.sum() >= 30 and 0 < y[m].sum() < m.sum() else np.nan)
            row["auc_delta"] = row["auc_banded"] - row["auc_baseline"]
            strata.append(row)
    by_stratum = pd.DataFrame(strata)
    by_stratum.round(4).to_csv(out / "auc_by_stratum.csv", index=False)
    say("")
    say(block(by_stratum.round(4), index=False))

    # ---------------------------------------------------------------- plot
    section("7. FIGURE")
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    _style(ax)
    for name, colour, style in ((BASELINE, ACCENT, "-"), (BANDED, WARM, "-")):
        fpr, tpr = curves[name]
        r = table[table["model"] == name].iloc[0]
        ax.plot(fpr, tpr, style, color=colour, linewidth=2.0, zorder=3,
                label=f"{labels[name]}\n   AUC {r['auc_roc']:.3f} "
                      f"[{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]")
    for name, colour in ((BASELINE, ACCENT), (BANDED, WARM)):
        r = next(o for o in op_rows if o["model"] == name)
        ax.scatter([r["operating_fpr"]], [r["operating_tpr"]], s=58, marker="o",
                   color=colour, zorder=5, edgecolor=SURFACE, linewidth=1.4)
    ax.annotate("argmax operating points:\nboth arms decide here",
                xy=(op_rows[0]["operating_fpr"], op_rows[0]["operating_tpr"]),
                xytext=(0.30, 0.48), fontsize=8.5, color=INK_MUTED,
                arrowprops={"arrowstyle": "-", "color": NEUTRAL, "linewidth": 1})
    ax.plot([0, 1], [0, 1], "--", color=NEUTRAL, linewidth=1.2, zorder=2,
            label="chance")
    ax.set_xlabel("False positive rate   (TD next visits called PIGD)")
    ax.set_ylabel("True positive rate   (PIGD next visits correctly called)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_title("ROC for the next observed subtype, TD vs PIGD\n"
                 f"{len(y):,} held-out predictions from {n_pat:,} patients; "
                 f"{int(y.sum()):,} PIGD, {int((~y).sum()):,} TD\n"
                 "the control arm's curve is identical to the baseline's",
                 fontsize=10, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    path = _finish(fig, out / "roc_td_vs_pigd.png")
    say(f"  wrote {path.name}")

    pts = [pd.DataFrame({"model": n, "fpr": f, "tpr": t})
           for n, (f, t) in curves.items()]
    pd.concat(pts, ignore_index=True).round(6).to_csv(
        out / "roc_points.csv", index=False)
    say(f"  wrote roc_points.csv ({sum(len(p) for p in pts):,} coordinates, so "
        "the figure can be audited without rerunning)")

    # ---------------------------------------------------------- provenance
    section("8. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "n_bootstrap": args.n_boot, "seed": 42, "source_run": RUN,
           "predictions_sha256": sha(src),
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<20} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
         "evaluation_set": {"n_cases": int(len(y)), "n_patients": n_pat,
                            "n_positive": int(y.sum()),
                            "positive_class": "PIGD", "score": "P_PIGD"},
         "auc": table.to_dict(orient="records"),
         "contrasts": contrasts,
         "operating_points": op_rows,
         "partial_auc_contrast": dp,
         "by_stratum": by_stratum.to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),
        encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
