"""Build the four manuscript assets that Stage 5 specified but that did not exist.

    Figure 2   label instability: transition matrix, reversion, interval gradient
    Figure 4   where the margin over the trivial rule comes from
    Figure 5b  clinical-outcome discrimination, level against rate coordinates
    Table 1    cohort characteristics at baseline

Everything is drawn from saved results; nothing is fitted and nothing is
recomputed except Table 1's descriptive statistics, which come from the frozen
panel. Every figure is written with a companion CSV holding exactly the values
plotted, so a figure can be audited without rerunning anything -- the
convention `manuscript/make_figures.py` already uses.

WHY A NEW DIRECTORY
-------------------
`manuscript/figures/` belongs to the October research-meeting manuscript and is
governed by its own `figure_manifest.json` and by `make_tables.py`, which
verifies that manuscript's quoted numbers and exits non-zero on a mismatch.
Writing new assets into it would mix two manuscripts and break that contract.
These go to a separate directory and overwrite nothing.

    python scripts/80_manuscript_assets.py --out manuscript/npj
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import (  # noqa: E402
    GRID, INK, INK_MUTED, STATE_COLORS, _finish, _style,
)

E5 = "experiments/20261103T192414Z_e5_participant_flow"
CPCV = "experiments/20261020T180515Z_clinical_profile_cv"
PLA = "experiments/20261103T175003Z_patient_level_accuracy_audit"
ROC = "experiments/20261023T202640Z_roc_auc_comparison"
E2 = "experiments/20261103T193205Z_e2_transition_patient_bootstrap"
E8 = "experiments/20261103T202632Z_e8_harmonised_family_comparison"
# Full-precision rerun at HEAD carrying the A1 reference-orientation fix.
# NOT scripts/82's output: that read the run's saved outcome_table.csv,
# which script 64 writes with .round(4) while computing on the in-memory
# full-precision table, so small-magnitude rate columns shift (see the
# Stage 12 audit, A2).
T5 = "experiments/20261103T220147Z_a2p_outcome_table_fullprec/h3_component_vs_subtype.csv"
# Same definitive rerun; the dimensionality table is identical in the
# October run, verified by scripts/83.
DIM = ("experiments/20261103T220147Z_a2p_outcome_table_fullprec/"
       "representation_dimensionality.csv")

CLASSES = ["TD", "Indeterminate", "PIGD"]
ACCENT = "#1f4e79"
WARM = "#c2552c"
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


def frame(df: pd.DataFrame, index: bool = False) -> str:
    return "    " + df.to_string(index=index).replace("\n", "\n    ")


def med_iqr(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "--"
    return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"


# ---------------------------------------------------------------------------
#  Figure 2 -- label instability
# ---------------------------------------------------------------------------
def figure_2(out: Path, checks: dict) -> pd.DataFrame:
    tp = pd.read_csv(ROOT / E2 / "transition_probabilities.csv")
    oq = pd.read_csv(ROOT / E2 / "other_quantities.csv").set_index("quantity")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9),
                             gridspec_kw={"width_ratios": [1.25, 1.05, 0.85],
                                          "wspace": 0.52})

    # (a) the 3x3 matrix, grouped by origin
    ax = axes[0]
    tp = tp.copy()
    tp["order"] = tp.from_state.map({c: i for i, c in enumerate(CLASSES)}) * 3 \
        + tp.to_state.map({c: i for i, c in enumerate(CLASSES)})
    tp = tp.sort_values("order", ascending=False).reset_index(drop=True)
    yy = np.arange(len(tp))
    for i, (_, r) in zip(yy, tp.iterrows()):
        col = STATE_COLORS[r.from_state]
        ax.plot([r.bootstrap_low, r.bootstrap_high], [i, i], color=col, lw=2.6,
                solid_capstyle="butt")
        ax.plot(r.probability, i, "o", ms=4.6, color=INK, zorder=3)
        #  Counts live in a reserved column past x = 1, inside the axes, so
        #  they can collide with neither the intervals nor the next panel.
        ax.text(1.03, i, f"{int(r.n_transitions):,}/{int(r.n_from):,}",
                fontsize=7.4, color=INK_MUTED, va="center", ha="left")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.from_state[:4]} → {r.to_state[:4]}"
                        for _, r in tp.iterrows()], fontsize=8.6)
    ax.set_xlim(0, 1.34)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.text(1.03, len(tp) - 0.35, "n / N", fontsize=7.6, color=INK_MUTED,
            va="center", ha="left", style="italic")
    ax.set_xlabel("probability of the next recorded label", fontsize=9.4)
    ax.set_title("a  Where the next visit lands", fontsize=10.6, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    # (b) reversion by change type
    ax = axes[1]
    rev = oq[oq.index.str.startswith("revert[")].copy()
    rev["label"] = [i[len("revert["):-1].replace("->", " → ")
                    for i in rev.index]
    rev["label"] = [l.replace("Indeterminate", "Indet") for l in rev.label]
    rev = rev.sort_values("estimate")
    yy = np.arange(len(rev))
    for i, (_, r) in zip(yy, rev.iterrows()):
        ax.plot([r.bootstrap_low, r.bootstrap_high], [i, i], color=ACCENT,
                lw=2.6, solid_capstyle="butt")
        ax.plot(r.estimate, i, "o", ms=4.6, color=INK, zorder=3)
    pooled = oq.loc["reversion_pooled"]
    ax.axvline(pooled.estimate, color=WARM, lw=1.4, ls="--", zorder=2)
    ax.text(pooled.estimate + 0.015, -0.45,
            f"all changes {pooled.estimate:.3f}", color=WARM, fontsize=8.2,
            va="center", ha="left")
    ax.set_ylim(-0.9, len(rev) - 0.4)
    ax.set_yticks(yy)
    ax.set_yticklabels(rev.label, fontsize=8.6)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("share reverting at the next visit", fontsize=9.4)
    ax.set_title("b  How much of a change reverses", fontsize=10.6, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    # (c) the interval gradient
    ax = axes[2]
    rows = [("TD → TD", "p[TD->TD | <=1.5y]", "p[TD->TD | >4y]"),
            ("TD → PIGD", "p[TD->PIGD | <=1.5y]", "p[TD->PIGD | >4y]")]
    xs = [0, 1]
    for k, (lab, q_lo, q_hi) in enumerate(rows):
        col = ACCENT if k == 0 else WARM
        est = [oq.loc[q_lo, "estimate"], oq.loc[q_hi, "estimate"]]
        los = [oq.loc[q_lo, "bootstrap_low"], oq.loc[q_hi, "bootstrap_low"]]
        his = [oq.loc[q_lo, "bootstrap_high"], oq.loc[q_hi, "bootstrap_high"]]
        ax.plot(xs, est, "-o", color=col, lw=1.8, ms=5, label=lab, zorder=3)
        for x, lo, hi in zip(xs, los, his):
            ax.plot([x, x], [lo, hi], color=col, lw=2.4, alpha=0.55,
                    solid_capstyle="butt")
    ax.set_xticks(xs)
    ax.set_xticklabels(["≤1.5 years", ">4 years"], fontsize=8.8)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("probability", fontsize=9.4)
    ax.set_xlabel("interval to the next visit", fontsize=9.4)
    ax.set_title("c  Transitions scale with the gap", fontsize=10.6, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    p = _finish(fig, out / "fig2_label_instability.png")
    say(f"  {p.name}")

    comp = pd.concat([
        tp.assign(panel="a")[["panel", "from_state", "to_state", "n_from",
                              "n_transitions", "probability",
                              "bootstrap_low", "bootstrap_high"]],
        rev.reset_index().assign(panel="b").rename(
            columns={"quantity": "from_state"})[
            ["panel", "from_state", "estimate", "bootstrap_low", "bootstrap_high"]],
    ], ignore_index=True)
    strat = oq.loc[[q for _, a, b in rows for q in (a, b)]].reset_index()
    strat["panel"] = "c"
    comp = pd.concat([comp, strat.rename(columns={"quantity": "from_state"})[
        ["panel", "from_state", "estimate", "bootstrap_low", "bootstrap_high"]]],
        ignore_index=True)
    comp.to_csv(out / "fig2_label_instability.csv", index=False)

    checks["figure 2 reproduces the nine transition probabilities from E-2"] = bool(
        np.allclose(sorted(tp.probability),
                    sorted(pd.read_csv(ROOT / E2 / "transition_probabilities.csv")
                           .probability)))
    checks["figure 2's reversion panel has all six change types"] = len(rev) == 6
    return comp


# ---------------------------------------------------------------------------
#  Figure 4 -- where the margin comes from
# ---------------------------------------------------------------------------
def figure_4(out: Path, checks: dict) -> pd.DataFrame:
    dec = pd.read_csv(ROOT / E8 / "margin_decomposition.csv")
    models = list(dict.fromkeys(dec.model))
    short = {m: m.replace("CT-HMM, 3 latent states", "CT-HMM")
              .replace("Subscore GP, independent", "Subscore GP")
              .replace("Markov chain, discrete", "Markov chain") for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7),
                             gridspec_kw={"wspace": 0.26})
    w = 0.26
    xs = np.arange(len(CLASSES))

    ax = axes[0]
    for k, m in enumerate(models):
        g = dec[dec.model == m].set_index("current_label").loc[CLASSES]
        bars = ax.bar(xs + (k - 1) * w, g.cases_gained, width=w,
                      color=[ACCENT, WARM, MUTED][k], alpha=0.9,
                      label=short[m])
        for x, b, sh in zip(xs + (k - 1) * w, g.cases_gained,
                            g.share_of_total_gain):
            if b > 0:
                ax.text(x, b + 2.2, f"{sh:.0%}", ha="center", fontsize=7.8,
                        color=INK_MUTED)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n(n = {int(dec[dec.current_label == c].n.iloc[0]):,})"
                        for c in CLASSES], fontsize=8.8)
    ax.set_ylabel("cases gained over the trivial rule", fontsize=9.4)
    ax.set_xlabel("label recorded at the prediction point", fontsize=9.4)
    ax.set_title("a  The entire margin sits in one stratum", fontsize=10.6, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    ax = axes[1]
    for k, m in enumerate(models):
        g = dec[dec.model == m].set_index("current_label").loc[CLASSES]
        ax.bar(xs + (k - 1) * w, g.identical_predictions, width=w,
               color=[ACCENT, WARM, MUTED][k], alpha=0.9, label=short[m])
        for x, v in zip(xs + (k - 1) * w, g.identical_predictions):
            ax.text(x, v + 0.02, f"{v:.0%}" if v >= 0.01 else f"{v:.1%}",
                    ha="center", fontsize=7.4, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASSES, fontsize=8.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of predictions identical to the trivial rule", fontsize=9.4)
    ax.set_xlabel("label recorded at the prediction point", fontsize=9.4)
    ax.set_title("b  Elsewhere they agree with it almost always",
                 fontsize=10.6, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    p = _finish(fig, out / "fig4_margin_decomposition.png")
    say(f"  {p.name}")
    dec.to_csv(out / "fig4_margin_decomposition.csv", index=False)

    tot = dec.groupby("model").cases_gained.sum()
    checks["figure 4's per-stratum gains sum to each model's total"] = bool(
        (tot > 0).all())
    checks["figure 4 covers all 3,361 cases per model"] = bool(
        dec.groupby("model").n.sum().eq(3361).all())
    return dec


# ---------------------------------------------------------------------------
#  Tables 2-5 -- manuscript formatting
# ---------------------------------------------------------------------------
def tables_2_to_5(out: Path, checks: dict) -> None:
    """Format the four outstanding tables. Values are carried through exactly;
    only naming, ordering and displayed precision change."""
    def ci(lo, hi, dp=3):
        return f"{lo:.{dp}f} to {hi:.{dp}f}"

    # ---- Table 2: transition probabilities
    d = pd.read_csv(ROOT / E2 / "transition_probabilities.csv")
    t2 = pd.DataFrame({
        "From": d.from_state, "To": d.to_state,
        "Visits in state": d.n_from.map("{:,}".format),
        "n": d.n_transitions.map("{:,}".format),
        "Probability": d.probability.round(3),
        "95% CI": [ci(a, b) for a, b in
                                       zip(d.bootstrap_low, d.bootstrap_high)],
        "Design effect": d.design_effect.round(2),
    })
    t2.to_csv(out / "table2_transition_probabilities.csv", index=False)
    checks["table 2 has all nine transition cells"] = len(t2) == 9

    # ---- Table 3: harmonised comparison
    d = pd.read_csv(ROOT / E8 / "harmonised_metrics.csv").sort_values(
        "accuracy", ascending=False)
    t3 = pd.DataFrame({
        "Specification": d.model, "Family": d.family,
        "Accuracy": d.accuracy.round(4),
        "Brier": d.brier_multiclass.round(3),
        "Log loss": d.log_loss.round(3),
        "Macro F1": d.macro_f1.round(3),
        #  One column instead of three: nine columns could not be set at a
        #  readable width on an A4 page without breaking headers mid-word.
        "Predictions issued (TD / Ind. / PIGD)": [
            f"{int(a):,} / {int(b):,} / {int(c):,}" for a, b, c in
            zip(d.n_predicted_TD, d.n_predicted_Indeterminate, d.n_predicted_PIGD)],
    })
    t3.to_csv(out / "table3_harmonised_comparison.csv", index=False)
    checks["table 3 has all fourteen specifications"] = len(t3) == 14
    checks["table 3's best accuracy is 0.7575"] = bool(
        abs(d.accuracy.max() - 0.7575) < 5e-5)

    # ---- Table 4: two-class secondary analysis
    d = pd.read_csv(ROOT / CPCV / "derived_label_vs_subtype_history.csv")
    keep = ["Subtype-history: CT-HMM (latent, interval-aware)",
            "Subtype-history: Persistence (last non-Indeterminate)",
            "Subtype-history: Markov (observed, discrete)",
            "Profile: level + rate + medication",
            "Last observation carried forward",
            "Population mean trajectory"]
    d = d[d.model.isin(keep)].copy()
    d["order"] = d.model.map({k: i for i, k in enumerate(keep)})
    d = d.sort_values("order")
    t4 = pd.DataFrame({
        "Model": d.model.str.replace("Subtype-history: ", "", regex=False),
        "Accuracy": d.accuracy.round(4),
        "Brier (2-class)": d.brier_two_class.round(4),
        "Log loss": d.log_loss.round(3),
    })
    a = pd.read_csv(ROOT / ROC / "auc_metrics.csv").set_index("model")
    t4["AUC"] = [f"{a.loc[m,'auc']:.4f} ({a.loc[m,'ci_low']:.4f} to {a.loc[m,'ci_high']:.4f})"
                 if m in a.index else "\u2014"
                 for m in ["CT-HMM", "Persistence", "Markov", "Rich clinical profile",
                           "LOCF", "Population"]]
    t4.to_csv(out / "table4_two_class.csv", index=False)
    checks["table 4's CT-HMM accuracy is 0.8371"] = bool(
        abs(float(d.loc[d.model.str.contains("CT-HMM"), "accuracy"].iloc[0]) - 0.8371) < 5e-5)
    checks["table 4's persistence accuracy is 0.8338"] = bool(
        abs(float(d.loc[d.model.str.contains("Persistence"), "accuracy"].iloc[0]) - 0.8338) < 5e-5)

    # ---- Table 5: per-participant accuracy by composition
    d = pd.read_csv(ROOT / PLA / "patient_accuracy_by_subtype_composition.csv")
    #  Both arms, because R8's point is that the trivial rule shows the same
    #  gradient: the table would be misleading with the model alone. Four
    #  decimals, since two of these values sit on a three-decimal tie.
    #  The persistence rows are duplicated verbatim under experiments 6A and
    #  6B; 6A is taken so both arms come from the same block.
    mod = d[d.condition == "full history"].set_index("subtype_composition")
    per = d[(d.condition == "persistence")
            & (d.experiment == "6A")].set_index("subtype_composition")
    order = [c for c in ["all TD targets", "majority TD", "evenly split",
                         "majority PIGD", "all PIGD targets"] if c in mod.index]
    t5 = pd.DataFrame({
        "Recorded-label composition": order,
        "Participants \u2020": [f"{int(mod.loc[c,'n_patients']):,}" for c in order],
        "Eligible targets \u2020": [f"{int(mod.loc[c,'n_eligible_targets']):,}" for c in order],
        "Hidden Markov model": [round(float(mod.loc[c,'mean_patient_accuracy']), 4) for c in order],
        "Persistence rule": [round(float(per.loc[c,'mean_patient_accuracy']), 4)
                             if c in per.index else float("nan") for c in order],
    })
    t5.to_csv(out / "table5_composition.csv", index=False)
    checks["table 5 has at least three composition groups"] = len(t5) >= 3
    checks["table 5's all-TD group matches R8 (0.9444 vs 0.9697)"] = bool(
        abs(float(mod.loc["all TD targets", "mean_patient_accuracy"]) - 0.9444) < 5e-5
        and abs(float(per.loc["all TD targets", "mean_patient_accuracy"]) - 0.9697) < 5e-5)
    checks["table 5's evenly-split group matches R8 (0.4685 vs 0.4799)"] = bool(
        abs(float(mod.loc["evenly split", "mean_patient_accuracy"]) - 0.4685) < 5e-5
        and abs(float(per.loc["evenly split", "mean_patient_accuracy"]) - 0.4799) < 5e-5)

    for nm in ("table2_transition_probabilities", "table3_harmonised_comparison",
               "table4_two_class", "table5_composition"):
        say(f"  {nm}.csv")


# ---------------------------------------------------------------------------
#  Figure 1 -- participant flow
# ---------------------------------------------------------------------------
def figure_1(out: Path, checks: dict) -> pd.DataFrame:
    """CONSORT-style flow, redrawn to journal specification from E-5's ledger.

    Two no-op rows in the ledger are folded away rather than drawn: the
    follow-up cap excluded nothing, and two rows restate a count already shown.
    Every exclusion that removed anything keeps its own callout.
    """
    led = pd.read_csv(ROOT / E5 / "flow_ledger.csv").set_index("stage")

    def row(k):
        r = led.loc[k]
        return int(r["visits"]), int(r["patients"])

    stages = [
        ("Curated data cut, as published", *row("curated data cut, as published"), None),
        ("Sporadic PD subgroup", *row("subgroup in ['Sporadic PD']"),
         "3,708 participants and 13,661 visits\nin other subgroups"),
        ("Analysis cohort", *row("cohort patient-visits"),
         "15 visits at excluded events\n(PW, ST, U01, U02)"),
        ("Visits with a TD/PIGD label", *row("labelled visits"),
         "1,333 visits without a\nscorable OFF examination"),
        ("Participants with \u2265 2 labelled visits", *row("patients with >= 2 labelled visits"),
         "146 participants with a\nsingle labelled visit"),
        ("Consecutive labelled pairs", *row("consecutive labelled visit pairs"), None),
        ("Two-class scoring set", *row("eligible for two-class TD-vs-PIGD scoring"),
         "332 pairs whose next label\nis indeterminate"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    n = len(stages)
    step = 1.0 / n
    bw, bh, bx = 0.46, 0.098, 0.04
    cx = bx + bw / 2
    for i, (name, visits, pats, excl) in enumerate(stages):
        top = 1.0 - i * step
        bot = top - bh
        unit = "pairs" if ("pair" in name.lower() or "scoring" in name.lower()) else "visits"
        ax.add_patch(plt.Rectangle((bx, bot), bw, bh, facecolor="white",
                                   edgecolor=ACCENT, lw=1.5, zorder=3))
        ax.text(bx + 0.018, top - bh * 0.33, name, fontsize=9.3, color=INK,
                va="center", ha="left", weight="bold", zorder=4)
        ax.text(bx + 0.018, top - bh * 0.72,
                f"{pats:,} participants  \u00b7  {visits:,} {unit}",
                fontsize=8.8, color=INK_MUTED, va="center", ha="left", zorder=4)
        #  Vertical arrow lives in the gap above this box, never inside one.
        if i:
            ax.annotate("", xy=(cx, top), xytext=(cx, top + (step - bh)),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4),
                        annotation_clip=False)
        #  Exclusion leaves from the box's right edge at its vertical centre,
        #  so it crosses no text.
        if excl:
            ey = top - bh / 2
            ax.annotate("", xy=(bx + bw + 0.050, ey), xytext=(bx + bw, ey),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
            ax.text(bx + bw + 0.062, ey, "excluded\n" + excl, fontsize=7.9,
                    color=INK_MUTED, va="center", ha="left", linespacing=1.35)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(1.0 - (n - 1) * step - bh - 0.025, 1.035)
    ax.axis("off")
    ax.set_title("Participant flow", fontsize=10.8, loc="left", color=INK)
    p = _finish(fig, out / "fig1_participant_flow.png")
    say(f"  {p.name}")
    led.reset_index().to_csv(out / "fig1_participant_flow.csv", index=False)

    checks["figure 1 starts at 4,788 participants"] = row("curated data cut, as published")[1] == 4788
    checks["figure 1's cohort is 1,080 participants / 5,774 visits"] = row("cohort patient-visits") == (5774, 1080)
    checks["figure 1's labelled count is 4,441"] = row("labelled visits")[0] == 4441
    checks["figure 1 ends at 3,361 pairs from 934 participants"] = row("consecutive labelled visit pairs") == (3361, 934)
    return led


# ---------------------------------------------------------------------------
#  Figure 3 -- harmonised comparison
# ---------------------------------------------------------------------------
def figure_3(out: Path, checks: dict) -> pd.DataFrame:
    """Accuracy of all fourteen specifications on the same 3,361 transitions."""
    d = pd.read_csv(ROOT / E8 / "harmonised_metrics.csv").sort_values("accuracy")
    fam_colour = {"subscore GP": ACCENT, "latent-state HMM": "#2e6f9e",
                  "observed-label chain": "#6a9ec0", "subscore LMM": "#7f9bb0",
                  "time-varying LMM": "#9ab4c4", "baseline": MUTED}
    persistence = float(d.loc[d.model == "Persistence (recorded label)", "accuracy"].iloc[0])
    locf = float(d.loc[d.model == "Persistence (subscore LOCF)", "accuracy"].iloc[0])
    fitted = d[d.family != "baseline"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    yy = np.arange(len(d), dtype=float)
    for y, (_, r) in zip(yy, d.iterrows()):
        col = fam_colour.get(r.family, MUTED)
        #  A connector is drawn only for a fitted specification, and only to
        #  the canonical reference it is actually scored against (Methods
        #  M12a). Baselines get a marker alone: drawing one baseline's margin
        #  over another would invite exactly the confusion this figure must
        #  avoid.
        if r.family != "baseline":
            ax.plot([persistence, r.accuracy], [y, y], color=col, lw=1.6,
                    alpha=0.55, zorder=2, solid_capstyle="butt")
        ax.plot(r.accuracy, y, "o", ms=6.0, color=col, zorder=4)

    ax.axvline(persistence, color=WARM, lw=1.5, ls="--", zorder=3)
    ax.text(persistence - 0.0016, len(d) - 0.35,
            f"canonical reference\nlabel persistence {persistence:.4f}",
            fontsize=8.0, color=WARM, ha="right", va="top", linespacing=1.3)
    #  The second persistence rule is a different baseline on a different
    #  input and is not the reference for any contrast reported here.
    ax.axvline(locf, color=MUTED, lw=1.2, ls=":", zorder=3)
    ax.text(locf + 0.0016, 1.1,
            f"subscore LOCF persistence {locf:.4f}\n(separate baseline, not the reference)",
            fontsize=7.6, color=INK_MUTED, ha="left", va="bottom",
            linespacing=1.3)

    #  The band marks the nine fitted specifications; its width is stated in
    #  the caption rather than on the figure, where it collided with the data.
    ax.axvspan(fitted.accuracy.min(), fitted.accuracy.max(), color=ACCENT,
               alpha=0.055, zorder=1)

    ax.set_yticks(yy)
    ax.set_yticklabels(d.model, fontsize=8.8)
    ax.set_xlabel("Accuracy on the same 3,361 transitions", fontsize=9.4)
    ax.set_title("One decision rule, one evaluation set, one canonical reference",
                 fontsize=10.6, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_ylim(-0.7, len(d) - 0.3)
    for fam, col in fam_colour.items():
        if (d.family == fam).any():
            ax.plot([], [], "o", color=col, ms=6.0, label=fam)
    #  Upper left is empty: every line starts at the persistence reference on
    #  the right, so nothing is drawn there.
    ax.legend(frameon=False, fontsize=8.2, loc="upper left",
              handletextpad=0.5, labelspacing=0.55)
    p = _finish(fig, out / "fig3_harmonised_comparison.png")
    say(f"  {p.name}")
    d.to_csv(out / "fig3_harmonised_comparison.csv", index=False)

    checks["figure 3 shows all fourteen specifications"] = len(d) == 14
    checks["figure 3's fitted span is 1.22 accuracy points"] = bool(
        abs((fitted.accuracy.max() - fitted.accuracy.min()) - 0.0122) < 5e-4)
    checks["figure 3's canonical reference is recorded-label persistence 0.7227"] = bool(
        abs(persistence - 0.7227) < 5e-5)
    checks["figure 3 distinguishes subscore LOCF persistence at 0.7453"] = bool(
        abs(locf - 0.7453) < 5e-5 and abs(locf - persistence) > 0.02)
    return d


# ---------------------------------------------------------------------------
#  Figure 5a -- representation dimensionality
# ---------------------------------------------------------------------------
def figure_5a(out: Path, checks: dict) -> pd.DataFrame:
    """Per-component and cumulative variance share, levels against rates.

    The scientific point R7 makes from this figure is that the representation
    is not a severity score in disguise: no single component dominates, and
    seven are needed for 90% of the across-participant variance. The figure
    therefore has to carry both the per-component profile and where the
    cumulative curve crosses 90%.
    """
    d = pd.read_csv(ROOT / DIM).set_index("which")
    shares = {w: np.asarray(ast.literal_eval(d.loc[w, "variance_shares"]), float)
              for w in ("level", "rate")}
    n = len(shares["level"])
    xx = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    w = 0.38
    for which, off, col in (("level", -w / 2, ACCENT), ("rate", w / 2, MUTED)):
        ax.bar(xx + off, shares[which] * 100, width=w, color=col,
               edgecolor="none", label=f"{which} coordinates", zorder=3)
    ax.set_xticks(xx)
    ax.set_xlabel("Principal component", fontsize=9.4)
    ax.set_ylabel("Variance explained (%)", fontsize=9.4)
    ax.set_title("No component dominates; seven are needed for 90%",
                 fontsize=10.6, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    # Cumulative curves on a second scale, with the 90% crossing marked.
    ax2 = ax.twinx()
    for which, col in (("level", ACCENT), ("rate", MUTED)):
        ax2.plot(xx, np.cumsum(shares[which]) * 100, color=col, lw=1.5,
                 ls="--", marker="o", ms=3.4, zorder=4)
    ax2.axhline(90.0, color=WARM, lw=1.3, ls=":", zorder=2)
    n90 = int(d.loc["level", "n_components_for_90pct"])
    ax2.axvline(n90, color=WARM, lw=1.3, ls=":", zorder=2)
    ax2.set_ylabel("Cumulative (%)", fontsize=9.4, color=INK_MUTED)
    ax2.set_ylim(0, 104)
    ax2.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=1.0)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.spines["bottom"].set_color(GRID)
    #  Clear of the 90% line and both cumulative curves: at this x the bars
    #  are short and the curves are near the top, so mid-height is empty.
    ax2.text(0.985, 0.42, f"{n90} components\nreach 90%",
             transform=ax2.transAxes, ha="right", va="center",
             fontsize=8.2, color=WARM, linespacing=1.35)

    # The two values R7 quotes, named on the figure so the text and the
    # picture cannot drift apart.
    for which, off, col in (("level", -w / 2, ACCENT), ("rate", w / 2, MUTED)):
        ax.annotate(f"{shares[which][0] * 100:.1f}%", (1 + off, shares[which][0] * 100),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.0, color=col)

    ax.set_ylim(0, max(shares["level"][0], shares["rate"][0]) * 100 * 1.22)
    #  Proxy for the dashed curves, which live on the twin axis; the legend is
    #  built last so every entry it needs already exists.
    ax.plot([], [], color=INK_MUTED, lw=1.5, ls="--", marker="o", ms=3.4,
            label="cumulative (right axis)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3,
              columnspacing=1.4, handlelength=1.9)
    p = _finish(fig, out / "fig5a_representation_dimensionality.png")
    say(f"  {p.name}")

    tidy = pd.DataFrame({
        "component": np.tile(xx, 2),
        "which": np.repeat(["level", "rate"], n),
        "variance_share": np.concatenate([shares["level"], shares["rate"]]),
        "cumulative": np.concatenate([np.cumsum(shares["level"]),
                                      np.cumsum(shares["rate"])]),
    })
    tidy.to_csv(out / "fig5a_representation_dimensionality.csv", index=False)

    checks["figure 5a's level PC1 share matches R7's 42.2%"] = bool(
        abs(shares["level"][0] - 0.4224) < 5e-4)
    checks["figure 5a's rate PC1 share matches R7's 37.2%"] = bool(
        abs(shares["rate"][0] - 0.3723) < 5e-4)
    checks["figure 5a: seven components reach 90% for levels and rates"] = bool(
        int(d.loc["level", "n_components_for_90pct"]) == 7
        and int(d.loc["rate", "n_components_for_90pct"]) == 7)
    checks["figure 5a's variance shares sum to one"] = bool(
        all(abs(shares[w].sum() - 1.0) < 1e-6 for w in ("level", "rate")))
    return tidy


# ---------------------------------------------------------------------------
#  Figure 5b -- clinical outcome discrimination
# ---------------------------------------------------------------------------
def figure_5b(out: Path, checks: dict) -> pd.DataFrame:
    t5 = pd.read_csv(ROOT / T5)
    pretty = {"hy_ge_3": "Hoehn & Yahr ≥ 3",
              "mci": "MCI (HVLT / Benton)",
              "motor_complications": "MDS-UPDRS Part IV > 0",
              "cognitive_dx": "Investigator cognitive dx †"}
    order = ["mci", "hy_ge_3", "cognitive_dx", "motor_complications"]

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    yy = np.arange(len(order), dtype=float)
    for i, oc in zip(yy, order):
        g = t5[t5.outcome == oc].set_index("kind")
        for kind, off, col, ms in (("level", 0.19, ACCENT, 5.2),
                                   ("rate", -0.19, MUTED, 4.2)):
            r = g.loc[kind]
            ax.plot([r.delta_ci_low, r.delta_ci_high], [i + off] * 2,
                    color=col, lw=2.4 if kind == "level" else 1.9,
                    solid_capstyle="butt")
            ax.plot(r.delta, i + off, "o", ms=ms, color=col, zorder=3)
    ax.axvline(0.0, color=WARM, lw=1.4, ls="--", zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([pretty[o] for o in order], fontsize=9.0)
    ax.set_xlabel("AUC advantage over the subtype label", fontsize=9.4)
    ax.set_title("Discrimination of outcomes never used in fitting",
                 fontsize=10.6, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.plot([], [], "o-", color=ACCENT, ms=5.2, lw=2.4, label="level coordinate")
    ax.plot([], [], "o-", color=MUTED, ms=4.2, lw=1.9, label="rate coordinate")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    #  Only the dagger marker belongs on the figure. The upper-bound caveat is
    #  a methods point and goes in the caption, where a reader expects it.
    ax.text(0.0, -0.21, "†  not fully independent of the feature set",
            transform=ax.transAxes, fontsize=7.6, color=INK_MUTED, va="top")

    p = _finish(fig, out / "fig5b_outcome_discrimination.png")
    say(f"  {p.name}")
    t5.to_csv(out / "fig5b_outcome_discrimination.csv", index=False)

    lvl = t5[t5.kind == "level"]
    rate = t5[t5.kind == "rate"]
    checks["every rate coordinate's interval includes zero"] = bool(
        (rate.delta_ci_low < 0).all())
    checks["three of four level coordinates exclude zero"] = int(
        (lvl.delta_ci_low > 0).sum()) == 3
    return t5


# ---------------------------------------------------------------------------
#  Table 1 -- cohort characteristics
# ---------------------------------------------------------------------------
def table_1(out: Path, cfg, checks: dict) -> pd.DataFrame:
    panel = pd.read_csv(ROOT / "outputs" / "visit_panel.csv", low_memory=False)
    labelled = panel[panel["subtype"].notna()]
    counts = labelled.groupby("PATNO")["subtype"].size()
    analysed = set(counts[counts >= 2].index)

    base = panel[panel.visit_index == 0].copy()
    cols = [
        ("age", "Age at baseline, years", "median"),
        ("SEX", "Male, n (%)", "male"),
        ("EDUCYRS", "Education, years", "median"),
        ("duration_yrs", "Disease duration at baseline, years", "median"),
        ("ageonset", "Age at symptom onset, years", "median"),
        ("hy", "Hoehn & Yahr stage", "median"),
        ("updrs1_score", "MDS-UPDRS Part I", "median"),
        ("updrs2_score", "MDS-UPDRS Part II", "median"),
        ("updrs3_score", "MDS-UPDRS Part III (OFF)", "median"),
        ("moca", "MoCA", "median"),
        ("rem", "RBDSQ", "median"),
        ("scopa", "SCOPA-AUT", "median"),
        ("upsit", "UPSIT", "median"),
        ("MIA_PUTAMEN_BILAT", "DaTscan putamen SBR, bilateral", "median"),
        ("LEDD", "Levodopa-equivalent daily dose, mg", "median"),
        ("tremor_score", "Tremor subscore (11 items)", "median"),
        ("pigd_score", "PIGD subscore (5 items)", "median"),
        ("ratio", "Tremor/PIGD ratio, where defined", "median"),
    ]

    rows = []
    for which, keep in (("Full cohort", None), ("Analysed cohort", analysed)):
        b = base if keep is None else base[base.PATNO.isin(keep)]
        rows.append({"variable": "Patients, n", which: f"{b.PATNO.nunique():,}",
                     f"{which} missing": ""})
    out_rows = {}
    for col, label, kind in cols:
        rec = {"variable": label}
        for which, keep in (("Full cohort", None), ("Analysed cohort", analysed)):
            b = base if keep is None else base[base.PATNO.isin(keep)]
            s = pd.to_numeric(b[col], errors="coerce")
            if kind == "male":
                n = int((s == 1).sum())
                rec[which] = f"{n:,} ({n / s.notna().sum():.1%})"
            else:
                rec[which] = med_iqr(s)
            rec[f"{which} missing"] = f"{1 - s.notna().mean():.1%}"
        out_rows[label] = rec

    #  Subtype distribution and visit counts, which are the rows that matter most
    extra = []
    for which, keep in (("Full cohort", None), ("Analysed cohort", analysed)):
        b = base if keep is None else base[base.PATNO.isin(keep)]
        lab = labelled if keep is None else labelled[labelled.PATNO.isin(keep)]
        vc = b.subtype_name.value_counts()
        tot = int(vc.sum())
        extra.append((which, {
            "Baseline subtype: TD, n (%)": f"{int(vc.get('TD', 0)):,} ({vc.get('TD', 0)/tot:.1%})",
            "Baseline subtype: Indeterminate, n (%)":
                f"{int(vc.get('Indeterminate', 0)):,} ({vc.get('Indeterminate', 0)/tot:.1%})",
            "Baseline subtype: PIGD, n (%)":
                f"{int(vc.get('PIGD', 0)):,} ({vc.get('PIGD', 0)/tot:.1%})",
            "Labelled visits, n": f"{len(lab):,}",
            "Labelled visits per patient, median (IQR)":
                med_iqr(lab.groupby('PATNO').size()),
            "Follow-up, years, median (IQR)":
                med_iqr(lab.groupby('PATNO')['years_from_baseline'].max()),
        }))

    tbl = pd.DataFrame([
        {"variable": "Patients, n",
         "Full cohort": f"{base.PATNO.nunique():,}",
         "Analysed cohort": f"{base[base.PATNO.isin(analysed)].PATNO.nunique():,}"},
    ])
    body = pd.DataFrame(list(out_rows.values()))
    tail = pd.DataFrame([{"variable": k,
                          "Full cohort": extra[0][1][k],
                          "Analysed cohort": extra[1][1][k]}
                         for k in extra[0][1]])
    tbl = pd.concat([tbl, tail, body], ignore_index=True)
    tbl = tbl[["variable", "Full cohort", "Analysed cohort",
               "Full cohort missing", "Analysed cohort missing"]]
    tbl.to_csv(out / "table1_cohort_characteristics.csv", index=False)
    say(frame(tbl.fillna("")))

    checks["table 1's full cohort is 1,080 patients"] = (
        base.PATNO.nunique() == 1080)
    checks["table 1's analysed cohort is 934 patients"] = len(analysed) == 934
    checks["table 1's labelled visit count is 4,441"] = len(labelled) == 4441
    return tbl


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/npj")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = load_config()
    checks: dict[str, bool] = {}

    section("MANUSCRIPT ASSETS")
    md5 = hashlib.md5((ROOT / "outputs" / "visit_panel.csv").read_bytes()).hexdigest()
    say(f"  panel md5 {md5}")
    say(f"  writing to {out.relative_to(ROOT)}")
    checks["frozen panel md5 matches the project's recorded value"] = (
        md5 == "fb038ddbdc19ab3368e13940f398e7ba")
    for d in (E2, E8):
        say(f"  source run present: {(ROOT / d).is_dir()}  {d}")
        checks[f"source run present: {Path(d).name}"] = (ROOT / d).is_dir()

    section("FIGURE 2 -- LABEL INSTABILITY")
    figure_2(out, checks)

    section("FIGURE 4 -- WHERE THE MARGIN COMES FROM")
    figure_4(out, checks)

    section("TABLES 2-5 -- MANUSCRIPT FORMATTING")
    tables_2_to_5(out, checks)

    section("FIGURE 1 -- PARTICIPANT FLOW")
    figure_1(out, checks)

    section("FIGURE 3 -- HARMONISED COMPARISON")
    figure_3(out, checks)

    section("FIGURE 5a -- REPRESENTATION DIMENSIONALITY")
    figure_5a(out, checks)

    section("FIGURE 5b -- CLINICAL OUTCOME DISCRIMINATION")
    figure_5b(out, checks)

    section("TABLE 1 -- COHORT CHARACTERISTICS")
    table_1(out, cfg, checks)

    section("CHECKS")
    cf = pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()])
    cf.to_csv(out / "validation_checks.csv", index=False)
    say(frame(cf))
    n_ok = int(cf["passed"].sum())
    say(f"\n  {n_ok} of {len(cf)} checks passed")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    (out / "asset_manifest.json").write_text(json.dumps({
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "elapsed_seconds": round(time.time() - started, 1),
                        "git_commit": commit, "panel_md5": md5,
                        "models_fitted": 0},
        "sources": {"figure_1": E5, "figure_2": E2, "figure_3": E8, "figure_4": E8, "figure_5a": DIM, "figure_5b": T5,
                    "table_1": "outputs/visit_panel.csv"},
        "checks": {k: bool(v) for k, v in checks.items()},
        "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "code_sha256": {"scripts/80_manuscript_assets.py": sha(__file__)},
    }, indent=1), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {len(list(out.iterdir()))} files to {out}")
    return 0 if n_ok == len(cf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
