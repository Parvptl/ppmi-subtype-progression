"""Do the forced TD/PIGD calls on Indeterminate visits track the nearer boundary?

The subtype label is Indeterminate when the tremor/PIGD ratio falls in the open
band (0.90, 1.15). Those visits are excluded from the project's binary scoring
because neither TD nor PIGD is the right answer. But each one still sits closer
to one boundary than the other, and if a model's forced call tracked that, the
prediction would carry information even where it cannot be scored.

This script tests exactly that, on held-out predictions only. Nothing is
refitted and neither model is modified.

The hypothesis is **not** assumed. It is tested against an explicit null -- the
agreement expected if the model's calls were independent of where in the band
the visit sits -- with patient-clustered intervals, and it is reported as
unsupported if that is what the numbers say.

    python scripts/67_indeterminate_boundary_analysis.py --out <dir>
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.clinical_profile import auc  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import (  # noqa: E402
    GRID, INK, INK_MUTED, STATE_COLORS, SURFACE, _finish, _style,
)

CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
RP_RUN = "experiments/20261020T180515Z_clinical_profile_cv"
CT_MODEL = "CT-HMM (latent, interval-aware)"
RP_MODEL = "Profile: level + rate + medication"

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


def _bootstrap_auc(score, target, patno, seed=42, n_boot=2000):
    """Percentile CI for AUC, resampling patients with replacement."""
    score = np.asarray(score, dtype=float)
    target = np.asarray(target).astype(bool)
    patno = np.asarray(patno)
    by: dict = {}
    for p, s_, t in zip(patno, score, target):
        by.setdefault(p, []).append((s_, t))
    patients = np.array(list(by.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for i in pick for r in by[patients[i]]]
        s_ = np.array([r[0] for r in rows]); t = np.array([r[1] for r in rows])
        if t.all() or not t.any():
            continue
        draws[b] = auc(s_, t)
    draws = draws[np.isfinite(draws)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def patient_bootstrap(values, patno, stat=np.mean, seed=42, n_boot=2000):
    """Percentile CI for a statistic, resampling patients with replacement."""
    values = np.asarray(values, dtype=float)
    patno = np.asarray(patno)
    by: dict = {}
    for p, v in zip(patno, values):
        by.setdefault(p, []).append(v)
    patients = np.array(list(by.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        draws[b] = stat(np.concatenate([by[patients[i]] for i in pick]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()
    LO, HI = cfg.subtype.pigd_threshold, cfg.subtype.td_threshold
    MID = (LO + HI) / 2

    # ------------------------------------------------------------ assemble
    section("1. THE ANALYSIS SET")
    say(f"  Indeterminate band: {LO} < ratio < {HI},  midpoint {MID:.4f}")

    panel = pd.read_csv(ROOT / "outputs/visit_panel.csv", low_memory=False)
    meas = panel[["PATNO", "EVENT_ID", "tremor_score", "pigd_score", "ratio",
                  "ratio_defined"]].copy()

    ct = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv", low_memory=False)
    ct = ct[ct["model"] == CT_MODEL]
    rp = pd.read_csv(ROOT / RP_RUN / "derived_label_predictions.csv", low_memory=False)
    rp = rp[rp["model"] == RP_MODEL]

    key = ["PATNO", "current_event", "target_event"]
    shared = (set(map(tuple, ct[key].to_numpy()))
              & set(map(tuple, rp[key].to_numpy())))

    def prep(d, tag):
        d = d[[tuple(r) in shared for r in d[key].to_numpy()]].copy()
        d = d[d["actual_next_label"] == "Indeterminate"]
        d = d.merge(meas, left_on=["PATNO", "target_event"],
                    right_on=["PATNO", "EVENT_ID"], how="left")
        return d.sort_values(key).reset_index(drop=True).add_prefix("")

    ct_i, rp_i = prep(ct, "ct"), prep(rp, "rp")
    assert (ct_i[key].to_numpy() == rp_i[key].to_numpy()).all(), "alignment"

    say(f"\n  Indeterminate-target cases shared by both models : {len(ct_i):,}")
    undef = int((~ct_i["ratio_defined"].astype(bool)).sum())
    say(f"    ratio undefined (PIGD subscore = 0)            : {undef}")
    say(f"      -> these are the 'both subscores zero' visits. No ratio exists,")
    say(f"         so no distance to a boundary exists either. EXCLUDED, counted.")

    ok = ct_i["ratio_defined"].astype(bool).to_numpy()
    ct_i, rp_i = ct_i[ok].reset_index(drop=True), rp_i[ok].reset_index(drop=True)
    ratio = ct_i["ratio"].to_numpy(float)
    patno = ct_i["PATNO"].to_numpy()
    say(f"    ANALYSED                                       : {len(ct_i):,}"
        f"  ({len(np.unique(patno)):,} patients)")

    # ------------------------------------------------- the ratio is discrete
    section("2. THE RATIO INSIDE THE BAND IS NEARLY BINARY")
    vc = pd.Series(ratio).round(6).value_counts().sort_index()
    dist = pd.DataFrame({"ratio": vc.index, "n": vc.to_numpy()})
    dist["share"] = dist["n"] / dist["n"].sum()
    dist["dist_to_PIGD_boundary"] = dist["ratio"] - LO
    dist["dist_to_TD_boundary"] = HI - dist["ratio"]
    dist["nearer"] = np.where(dist["dist_to_PIGD_boundary"]
                              < dist["dist_to_TD_boundary"], "PIGD", "TD")
    dist.round(4).to_csv(out / "ratio_values.csv", index=False)
    say(block(dist.round(4), index=False))
    say("")
    say("  The tremor score is a mean of 11 integer items and the PIGD score a")
    say("  mean of 5, so inside a band only 0.25 wide the ratio can take very")
    say("  few values. Most visits sit a hundredth or two from a boundary, not")
    say("  in the middle of the band. 'Distance to the nearer boundary' is")
    say("  therefore close to a binary label, not a continuous quantity.")

    nearer = np.where(ratio < MID, "PIGD", "TD")
    say(f"\n  nearer the PIGD boundary : {(nearer=='PIGD').sum():>4}"
        f"  ({(nearer=='PIGD').mean():.1%})")
    say(f"  nearer the TD boundary   : {(nearer=='TD').sum():>4}"
        f"  ({(nearer=='TD').mean():.1%})")

    # ------------------------------------------------------------- the test
    section("3. DOES EACH MODEL'S FORCED CALL TRACK THE NEARER BOUNDARY?")
    rows, per_model = [], {}
    for name, d in (("CT-HMM", ct_i), ("Rich clinical profile", rp_i)):
        pred = d["predicted_state"].to_numpy()
        agree = (pred == nearer).astype(float)
        lo, hi = patient_bootstrap(agree, patno, n_boot=args.n_boot)

        # Null: the agreement expected if calls were independent of position.
        p_pred_pigd = float((pred == "PIGD").mean())
        p_near_pigd = float((nearer == "PIGD").mean())
        expected = p_pred_pigd * p_near_pigd + (1 - p_pred_pigd) * (1 - p_near_pigd)
        excess = agree.mean() - expected
        elo, ehi = patient_bootstrap(agree - expected, patno, n_boot=args.n_boot)

        # Threshold-free: does P_PIGD rank nearer-PIGD cases higher? The
        # agreement test above reads a single threshold and is dominated by the
        # fact that both models mostly say TD, so a ranking measure is the
        # fairer test of whether any signal is present at all.
        score = d["P_PIGD"].to_numpy(float)
        target = (nearer == "PIGD")
        a = auc(score, target)
        a_lo, a_hi = _bootstrap_auc(score, target, patno, n_boot=args.n_boot)

        per_model[name] = {"pred": pred, "agree": agree}
        rows.append({
            "model": name, "n": len(d), "n_patients": int(len(np.unique(patno))),
            "predicted_PIGD": int((pred == "PIGD").sum()),
            "predicted_TD": int((pred == "TD").sum()),
            "share_predicted_PIGD": p_pred_pigd,
            "agreement_with_nearer_boundary": float(agree.mean()),
            "agreement_ci_low": lo, "agreement_ci_high": hi,
            "expected_under_independence": expected,
            "excess_over_null": float(excess),
            "excess_ci_low": elo, "excess_ci_high": ehi,
            "auc_P_PIGD_for_nearer_PIGD": a,
            "auc_ci_low": a_lo, "auc_ci_high": a_hi,
        })
        say(f"\n  {name}")
        say(f"    predicted PIGD {int((pred=='PIGD').sum()):>4} / "
            f"TD {int((pred=='TD').sum()):>4}   "
            f"({p_pred_pigd:.1%} PIGD)")
        say(f"    agreement with the nearer boundary : {agree.mean():.4f} "
            f"[{lo:.4f}, {hi:.4f}]")
        say(f"    expected if calls were independent : {expected:.4f}")
        say(f"    EXCESS over that null              : {excess:+.4f} "
            f"[{elo:+.4f}, {ehi:+.4f}]")
        verdict = ("excludes zero -> the calls DO track the boundary"
                   if elo > 0 else
                   ("excludes zero on the wrong side -> calls track it INVERSELY"
                    if ehi < 0 else
                    "INCLUDES ZERO -> no evidence the calls track the boundary"))
        say(f"    -> {verdict}")
        say(f"    AUC of P_PIGD for 'nearer PIGD'    : {a:.4f} "
            f"[{a_lo:.4f}, {a_hi:.4f}]")
        say(f"    -> {'excludes 0.5 -> weak but real ranking signal' if a_lo > 0.5 else 'INCLUDES 0.5 -> no ranking signal either'}")

    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "boundary_agreement.csv", index=False)

    section("4. SENSITIVITY: ONLY VISITS GENUINELY CLOSE TO A BOUNDARY")
    say("  The midpoint rule labels a visit at ratio 1.0227 'nearer PIGD' by a")
    say("  margin of 0.0046, which is noise rather than a meaningful position.")
    say("  Restricting to visits within 0.05 of a boundary tests the hypothesis")
    say("  where the construct actually means something.")

    margin = np.minimum(ratio - LO, HI - ratio)
    close = margin < 0.05
    say(f"\n  visits within 0.05 of a boundary : {int(close.sum())} of {len(ratio)}")
    say(f"    nearer PIGD {int((nearer[close]=='PIGD').sum())}"
        f"   nearer TD {int((nearer[close]=='TD').sum())}")

    sens = []
    if close.sum() >= 30 and len(set(nearer[close])) == 2:
        for name, d in (("CT-HMM", ct_i), ("Rich clinical profile", rp_i)):
            pred = d["predicted_state"].to_numpy()[close]
            nb = nearer[close]
            ag = (pred == nb).astype(float)
            lo2, hi2 = patient_bootstrap(ag, patno[close], n_boot=args.n_boot)
            pp = float((pred == "PIGD").mean())
            pn = float((nb == "PIGD").mean())
            exp2 = pp * pn + (1 - pp) * (1 - pn)
            elo2, ehi2 = patient_bootstrap(ag - exp2, patno[close],
                                           n_boot=args.n_boot)
            sens.append({"model": name, "n": int(close.sum()),
                         "agreement": float(ag.mean()),
                         "ci_low": lo2, "ci_high": hi2,
                         "expected_under_independence": exp2,
                         "excess": float(ag.mean() - exp2),
                         "excess_ci_low": elo2, "excess_ci_high": ehi2})
            say(f"\n  {name}: agreement {ag.mean():.4f} [{lo2:.4f}, {hi2:.4f}]"
                f"   excess {ag.mean()-exp2:+.4f} [{elo2:+.4f}, {ehi2:+.4f}]")
            say(f"    -> {'excludes zero' if elo2 > 0 else 'INCLUDES ZERO'}")
        pd.DataFrame(sens).round(4).to_csv(out / "sensitivity_close_to_boundary.csv",
                                           index=False)

    section("5. COUNTS AND PERCENTAGES")
    cross = []
    for name, d in (("CT-HMM", ct_i), ("Rich clinical profile", rp_i)):
        pred = d["predicted_state"].to_numpy()
        for nb in ("PIGD", "TD"):
            m = nearer == nb
            for pc in ("PIGD", "TD"):
                n = int((m & (pred == pc)).sum())
                cross.append({"model": name, "nearer_boundary": nb,
                              "predicted": pc, "n": n,
                              "pct_within_nearer_group": n / max(m.sum(), 1),
                              "matches_nearer": nb == pc})
    cross_df = pd.DataFrame(cross)
    cross_df.round(4).to_csv(out / "counts_and_percentages.csv", index=False)
    say(block(cross_df.round(4), index=False))
    say("")
    say("  Read the rows where matches_nearer is True: those are the calls that")
    say("  agree with the nearer boundary.")

    # ------------------------------------------------------------- figures
    section("6. FIGURES")
    # (a) ratio distribution with both boundaries
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    _style(ax)
    colours = [STATE_COLORS["PIGD"] if r < MID else STATE_COLORS["TD"]
               for r in dist["ratio"]]
    ax.bar(dist["ratio"], dist["n"], width=0.004, color=colours, zorder=3)
    for x, lab, col in ((LO, f"PIGD boundary\nratio = {LO}", STATE_COLORS["PIGD"]),
                        (HI, f"TD boundary\nratio = {HI}", STATE_COLORS["TD"])):
        ax.axvline(x, color=col, linewidth=2.0, zorder=2)
        ax.text(x, ax.get_ylim()[1] * 0.96, lab, rotation=0, ha="center",
                va="top", fontsize=8.5, color=col,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.axvline(MID, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(MID, ax.get_ylim()[1] * 0.55, f"  midpoint {MID}", fontsize=8,
            color=INK_MUTED)
    for _, r in dist.iterrows():
        ax.text(r["ratio"], r["n"] + 3, f"{int(r['n'])}", ha="center",
                fontsize=8, color=INK)
    ax.set_xlabel("Actual tremor / PIGD ratio at the next visit")
    ax.set_ylabel("Visits")
    ax.set_xlim(LO - 0.03, HI + 0.03)
    ax.set_title("Where Indeterminate visits actually sit inside the band\n"
                 f"{len(ct_i):,} held-out Indeterminate targets with a defined "
                 f"ratio; bars coloured by the nearer boundary",
                 fontsize=11, loc="left")
    ax.legend(handles=[Patch(facecolor=STATE_COLORS["PIGD"],
                             label=f"nearer the PIGD boundary (n={(nearer=='PIGD').sum()})"),
                       Patch(facecolor=STATE_COLORS["TD"],
                             label=f"nearer the TD boundary (n={(nearer=='TD').sum()})")],
              frameon=False, fontsize=9, loc="upper center")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    p1 = _finish(fig, out / "fig_ratio_distribution.png")
    say(f"  wrote {p1.name}")

    # (b) model comparison
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    _style(ax)
    x = np.arange(2)
    w = 0.36
    for k, (name, colour) in enumerate((("CT-HMM", ACCENT),
                                        ("Rich clinical profile", COOL))):
        pred = per_model[name]["pred"]
        vals = [float((pred[nearer == nb] == "PIGD").mean()) for nb in ("PIGD", "TD")]
        ax.bar(x + k * w - w / 2, vals, w * 0.92, color=colour, label=name, zorder=3)
        for xi, v in zip(x + k * w - w / 2, vals):
            ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"nearer PIGD boundary\n(n={(nearer=='PIGD').sum()})",
                        f"nearer TD boundary\n(n={(nearer=='TD').sum()})"])
    ax.set_ylabel("Share of calls that were PIGD")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Does the call shift with the nearer boundary?\n"
                 "if it tracked, the left bar would exceed the right",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    _style(ax)
    y = np.arange(len(table))
    ax.barh(y, table["agreement_with_nearer_boundary"],
            color=[ACCENT, COOL], height=0.5, zorder=3)
    ax.errorbar(table["agreement_with_nearer_boundary"], y,
                xerr=[table["agreement_with_nearer_boundary"] - table["agreement_ci_low"],
                      table["agreement_ci_high"] - table["agreement_with_nearer_boundary"]],
                fmt="none", ecolor=INK_MUTED, elinewidth=1.4, capsize=4, zorder=4)
    for yi, (_, r) in zip(y, table.iterrows()):
        ax.axvline(r["expected_under_independence"], color=WARM,
                   linestyle="--", linewidth=1.6, zorder=2,
                   label="expected if independent" if yi == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels(table["model"], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Agreement with the nearer boundary (95% CI, patients resampled)")
    ax.set_title("(b) Agreement against the independence null", fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Forced TD/PIGD calls on visits whose actual label is "
                 "Indeterminate", fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p2 = _finish(fig, out / "fig_model_comparison.png")
    say(f"  wrote {p2.name}")

    # ---------------------------------------------------------- provenance
    section("7. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "n_bootstrap": args.n_boot, "pigd_threshold": LO,
           "td_threshold": HI, "midpoint": MID,
           "ct_hmm_run": CT_RUN, "profile_run": RP_RUN,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<18} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
         "n_excluded_undefined_ratio": undef,
         "ratio_values": dist.to_dict(orient="records"),
         "agreement": table.to_dict(orient="records"),
         "counts": cross_df.to_dict(orient="records"),
         "sensitivity_close_to_boundary": sens},
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
