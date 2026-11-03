"""ROC curves and AUC for TD-vs-PIGD, both methods, on identical held-out cases.

Reuses the two existing experiment directories. Nothing is refitted and neither
model is touched: this reads the held-out probabilities they already produced.

AUC is threshold-free, so unlike accuracy it does not depend on the argmax
decision rule, and unlike Brier it is invariant to calibration. It measures one
thing only -- how well each model *ranks* a PIGD next visit above a TD one.

Confidence intervals resample **patients**, not predictions: a patient with
fourteen visits contributes fourteen correlated rows, and a row-level bootstrap
would understate the uncertainty.

    python scripts/66_roc_auc_comparison.py --out <dir>
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
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.clinical_profile import auc as manual_auc  # noqa: E402
from ppmi_subtype.plots import (  # noqa: E402
    GRID, INK, INK_MUTED, SURFACE, _finish, _style,
)

CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
RP_RUN = "experiments/20261020T180515Z_clinical_profile_cv"
CT_MODEL = "CT-HMM (latent, interval-aware)"
RP_MODEL = "Profile: level + rate + medication"
PERSIST = "Persistence (last non-Indeterminate)"

ACCENT = "#1f4e79"
COOL = "#2a78d6"
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


def patient_bootstrap_auc(score, positive, patno, seed=42, n_boot=2000):
    """Percentile CI for AUC, resampling patients with replacement."""
    score = np.asarray(score, float)
    positive = np.asarray(positive).astype(bool)
    patno = np.asarray(patno)

    by_patient: dict = {}
    for p, s, y in zip(patno, score, positive):
        by_patient.setdefault(p, []).append((s, y))
    patients = np.array(list(by_patient.keys()), dtype=object)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    draws[:] = np.nan
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for i in pick for r in by_patient[patients[i]]]
        s = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        if y.all() or not y.any():
            continue
        draws[b] = roc_auc_score(y, s)
    draws = draws[np.isfinite(draws)]
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
            int(len(draws)))


def patient_bootstrap_auc_diff(sa, sb, positive, patno, seed=42, n_boot=2000):
    """Percentile CI for AUC(a) - AUC(b), paired, resampling patients."""
    sa, sb = np.asarray(sa, float), np.asarray(sb, float)
    positive = np.asarray(positive).astype(bool)
    patno = np.asarray(patno)

    by_patient: dict = {}
    for p, a, b_, y in zip(patno, sa, sb, positive):
        by_patient.setdefault(p, []).append((a, b_, y))
    patients = np.array(list(by_patient.keys()), dtype=object)

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for k in pick for r in by_patient[patients[k]]]
        a = np.array([r[0] for r in rows]); b_ = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows])
        if y.all() or not y.any():
            continue
        draws[i] = roc_auc_score(y, a) - roc_auc_score(y, b_)
    draws = draws[np.isfinite(draws)]
    return {"delta": float(roc_auc_score(positive, sa)
                           - roc_auc_score(positive, sb)),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "n_draws": int(len(draws))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ------------------------------------------------------------- matching
    section("1. BUILDING THE SHARED EVALUATION SET")
    ct = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv", low_memory=False)
    rp = pd.read_csv(ROOT / RP_RUN / "derived_label_predictions.csv",
                     low_memory=False)
    persist = ct[ct["model"] == PERSIST].copy()
    ct = ct[ct["model"] == CT_MODEL].copy()
    rp = rp[rp["model"] == RP_MODEL].copy()
    say(f"  CT-HMM predictions       {len(ct):,}")
    say(f"  rich-profile predictions {len(rp):,}")

    key = ["PATNO", "current_event", "target_event"]
    keep = (set(map(tuple, ct[key].to_numpy()))
            & set(map(tuple, rp[key].to_numpy())))
    say(f"  shared visit pairs       {len(keep):,}")

    def restrict(d):
        d = d[[tuple(r) in keep for r in d[key].to_numpy()]].copy()
        d = d[d["eligible"] & d["two_class_defined"]]
        return d.sort_values(key).reset_index(drop=True)

    ct, rp, persist = restrict(ct), restrict(rp), restrict(persist)
    assert (ct[key].to_numpy() == rp[key].to_numpy()).all(), "row alignment"
    assert (ct[key].to_numpy() == persist[key].to_numpy()).all(), "row alignment"

    y = (ct["actual_next_label"] == "PIGD").to_numpy()
    y_rp = (rp["actual_next_label"] == "PIGD").to_numpy()
    assert (y == y_rp).all(), "the two tables disagree on the actual label"
    patno = ct["PATNO"].to_numpy()

    say(f"\n  eligible, both models, identical rows: {len(ct):,}")
    say(f"    patients            {len(np.unique(patno)):,}")
    say(f"    actual PIGD         {int(y.sum()):,}  ({y.mean():.1%})")
    say(f"    actual TD           {int((~y).sum()):,}")
    say("  Indeterminate targets are excluded from this binary evaluation, as")
    say("  everywhere else in the project; they are not scored for either model.")

    # ----------------------------------------------------------------- ROC
    section("2. ROC AND AUC")
    models = {
        "CT-HMM": (ct["P_PIGD"].to_numpy(float), ACCENT, "-"),
        "Rich clinical profile": (rp["P_PIGD"].to_numpy(float), COOL, "-"),
    }
    rows, curves = [], {}
    for name, (score, colour, style) in models.items():
        a = roc_auc_score(y, score)
        a_manual = manual_auc(score, y)
        if not np.isclose(a, a_manual, atol=1e-9):
            raise SystemExit(f"AUC disagreement for {name}: {a} vs {a_manual}")
        lo, hi, nd = patient_bootstrap_auc(score, y, patno, n_boot=args.n_boot)
        fpr, tpr, thr = roc_curve(y, score)
        curves[name] = (fpr, tpr, colour, style)
        rows.append({"model": name, "n": len(y), "n_patients": int(len(np.unique(patno))),
                     "n_positive": int(y.sum()), "auc": a,
                     "ci_low": lo, "ci_high": hi, "n_bootstrap": nd})
        say(f"  {name:<24} AUC {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
            f"   (cross-checked against the project's own AUC: OK)")

    # Persistence emits 0/1 only, so its ROC is a single interior point.
    ps = persist["P_PIGD"].to_numpy(float)
    a_p = roc_auc_score(y, ps)
    lo_p, hi_p, _ = patient_bootstrap_auc(ps, y, patno, n_boot=args.n_boot)
    tpr_p = float(ps[y].mean()); fpr_p = float(ps[~y].mean())
    rows.append({"model": "Persistence rule (reference)", "n": len(y),
                 "n_patients": int(len(np.unique(patno))),
                 "n_positive": int(y.sum()), "auc": a_p,
                 "ci_low": lo_p, "ci_high": hi_p, "n_bootstrap": args.n_boot})
    say(f"  {'Persistence rule':<24} AUC {a_p:.4f}  95% CI [{lo_p:.4f}, {hi_p:.4f}]")
    say("    its probabilities are 0/1, so it has a single operating point, not")
    say("    a curve; its AUC is bounded by that and is shown for reference only.")

    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "auc_metrics.csv", index=False)

    section("3. PAIRED DIFFERENCE")
    d = patient_bootstrap_auc_diff(rp["P_PIGD"].to_numpy(float),
                                   ct["P_PIGD"].to_numpy(float), y, patno,
                                   n_boot=args.n_boot)
    say(f"  rich profile - CT-HMM : {d['delta']:+.4f}"
        f"  95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    verdict = ("includes zero -> no significant difference"
               if d["ci_low"] <= 0 <= d["ci_high"] else "excludes zero")
    say(f"  -> the interval {verdict}")
    pd.DataFrame([{"contrast": "rich profile - CT-HMM", **d}]).round(4).to_csv(
        out / "auc_contrast.csv", index=False)

    section("4. AUC BY THE LABEL RECORDED AT THE PREDICTION POINT")
    sub = []
    for cur in ("TD", "Indeterminate", "PIGD"):
        m = (ct["current_label"] == cur).to_numpy()
        if m.sum() < 30 or y[m].all() or not y[m].any():
            sub.append({"current_label": cur, "n": int(m.sum()),
                        "ct_hmm_auc": np.nan, "profile_auc": np.nan})
            continue
        sub.append({"current_label": cur, "n": int(m.sum()),
                    "n_positive": int(y[m].sum()),
                    "ct_hmm_auc": roc_auc_score(y[m], ct["P_PIGD"].to_numpy()[m]),
                    "profile_auc": roc_auc_score(y[m], rp["P_PIGD"].to_numpy()[m])})
    by_cur = pd.DataFrame(sub)
    by_cur.round(4).to_csv(out / "auc_by_current_label.csv", index=False)
    say(block(by_cur.round(4), index=False))
    say("")
    say("  Within a single current label the ranking problem is much harder,")
    say("  because the strongest predictor -- the current label itself -- is")
    say("  held constant. These are not comparable to the pooled AUC above.")

    # ---------------------------------------------------------------- plot
    section("5. FIGURE")
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    _style(ax)
    for name, (fpr, tpr, colour, style) in curves.items():
        r = table[table["model"] == name].iloc[0]
        ax.plot(fpr, tpr, style, color=colour, linewidth=2.0, zorder=3,
                label=f"{name}\n   AUC {r['auc']:.3f} "
                      f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    ax.scatter([fpr_p], [tpr_p], s=70, marker="D", color=WARM, zorder=4,
               edgecolor=SURFACE, linewidth=1.2,
               label=f"Persistence rule (single operating point)\n   AUC {a_p:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=NEUTRAL, linewidth=1.2, zorder=2,
            label="chance")

    ax.set_xlabel("False positive rate   (TD next visits called PIGD)")
    ax.set_ylabel("True positive rate   (PIGD next visits correctly called)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_title("ROC for the next observed subtype, TD vs PIGD\n"
                 f"{len(y):,} held-out predictions from "
                 f"{len(np.unique(patno)):,} patients; "
                 f"{int(y.sum()):,} PIGD, {int((~y).sum()):,} TD",
                 fontsize=11, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    path = _finish(fig, out / "roc_td_vs_pigd.png")
    say(f"  wrote {path.name}")

    # Curve coordinates, so the figure can be audited without rerunning.
    pts = []
    for name, (fpr, tpr, _, _) in curves.items():
        pts.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
    pd.concat(pts, ignore_index=True).to_csv(out / "roc_points.csv", index=False)

    # ---------------------------------------------------------- provenance
    section("6. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "n_bootstrap": args.n_boot, "ct_hmm_run": CT_RUN,
           "profile_run": RP_RUN,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<16} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
         "auc": table.to_dict(orient="records"),
         "contrast": d,
         "auc_by_current_label": by_cur.to_dict(orient="records")},
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
