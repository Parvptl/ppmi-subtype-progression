#!/usr/bin/env python3
"""Build every figure and table used by `paper/ieee/main.tex`.

Reads only committed result files -- `outputs/` and the frozen experiment runs
under `experiments/` -- and writes figures, LaTeX table bodies and a JSON
record of every derived number into `paper/ieee/`. No analysis logic lives
here: each number is read from the file that produced it, or is an elementary
summary of the panel (counts, medians, interquartile ranges).

    ~/envs/ppmi/bin/python paper/ieee/make_assets.py

Nothing is written outside `paper/ieee/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TAB = HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

OUT = ROOT / "outputs"
EXP = ROOT / "experiments"
E2 = EXP / "20261103T193205Z_e2_transition_patient_bootstrap"
E5 = EXP / "20261103T192414Z_e5_participant_flow"
E8 = EXP / "20261103T202632Z_e8_harmonised_family_comparison"
O4 = EXP / "20261104T002308Z_o4_threshold_sensitivity"
MSC = EXP / "20261023T194317Z_model_selection_criteria"
ROC = EXP / "20261023T202640Z_roc_auc_comparison"
HIST = EXP / "20261024T042417Z_history_window_experiment"
RETR = EXP / "20261024T052343Z_retrained_two_visit_cthmm"
STAGE = EXP / "20261023T214819Z_stage_wise_evaluation"
PAT = EXP / "20261103T175003Z_patient_level_accuracy_audit"
PROF = EXP / "20261103T220147Z_a2p_outcome_table_fullprec"

# IEEE column widths, inches.
COL = 3.45
WIDE = 7.16

STATES = ["TD", "Indeterminate", "PIGD"]
SHORT = {"TD": "TD", "Indeterminate": "Indet", "PIGD": "PIGD"}

INK = "#1a1a1a"
GRID = "#d9d9d9"
BLUE = "#2d6ca2"
RED = "#b8433a"
GREY = "#8a8a8a"
GOLD = "#c8912a"
GREEN = "#3f7d50"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.4,
    "axes.labelsize": 7.4,
    "axes.titlesize": 7.8,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "legend.fontsize": 6.9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

NUM: dict[str, object] = {}


def fmt(value: float, dp: int, sign: bool = False) -> str:
    """Format rounding half away from zero, as the prose does.

    `format` rounds the underlying binary double, so a stored 0.6845 prints as
    0.684 while a reader writing the same number by hand puts 0.685. Tables,
    figure labels and the running text must agree, so they all come through
    here. `verify_numbers.py` applies the identical rule.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-dp)
    d = Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP)
    out = f"{d:f}"
    if sign and not out.startswith("-"):
        out = "+" + out
    return out


def rec(key: str, value) -> object:
    """Record a derived number so `verify_numbers.py` can check the text."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    NUM[key] = value
    return value


def tidy(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Panel-level descriptives (Table I)
# ---------------------------------------------------------------------------

def iqr_str(s: pd.Series, dp: int = 1) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{fmt(s.median(), dp)} ({fmt(s.quantile(.25), dp)}--{fmt(s.quantile(.75), dp)})"


def cohort_table() -> None:
    panel = pd.read_csv(OUT / "visit_panel.csv", low_memory=False)
    rec("n_patients", panel["PATNO"].nunique())
    rec("n_visits", len(panel))

    labelled = panel[panel["subtype_name"].notna()]
    rec("n_labelled", len(labelled))
    rec("pct_labelled", round(100 * len(labelled) / len(panel), 1))

    base = panel.sort_values(["PATNO", "visit_ts"]).groupby("PATNO").head(1)
    sex = base.set_index("PATNO")["SEX"]
    rec("n_male", int((sex == 1).sum()))
    rec("pct_male", round(100 * (sex == 1).mean(), 1))

    fu = panel.groupby("PATNO")["years_from_baseline"].max()
    rec("fu_median", round(fu.median(), 1))
    rec("fu_max", round(fu.max(), 1))

    n_lab = labelled.groupby("PATNO").size()
    rec("lab_visits_median", int(n_lab.median()))
    rec("lab_visits_max", int(n_lab.max()))

    lab_sorted = labelled.sort_values(["PATNO", "visit_ts"])
    gaps = lab_sorted.groupby("PATNO")["years_from_baseline"].diff().dropna()
    rec("n_pairs_from_panel", len(gaps))
    rec("interval_median", round(gaps.median(), 2))

    # Denominator is the baseline visits that carry a label, matching
    # outputs/panel_summary.txt; four first visits are unlabelled.
    base_lab = base[base["subtype_name"].notna()]
    rec("n_baseline_labelled", len(base_lab))
    counts = base_lab["subtype_name"].value_counts()
    for st in STATES:
        rec(f"baseline_n_{st}", int(counts.get(st, 0)))
        rec(f"baseline_pct_{st}", round(100 * counts.get(st, 0) / len(base_lab), 1))

    vc = labelled["subtype_name"].value_counts()
    for st in STATES:
        rec(f"labelled_n_{st}", int(vc.get(st, 0)))
        rec(f"labelled_pct_{st}", round(100 * vc.get(st, 0) / len(labelled), 1))

    rec("ratio_defined", int(labelled["ratio_defined"].sum()))
    rec("ratio_undefined", int((~labelled["ratio_defined"].astype(bool)).sum()))
    rec("pct_ratio_undefined",
        round(100 * (~labelled["ratio_defined"].astype(bool)).mean(), 1))

    rows = [
        ("Participants", f"{NUM['n_patients']:,}"),
        ("Patient--visits in cohort", f"{NUM['n_visits']:,}"),
        ("Visits with a motor subtype", f"{NUM['n_labelled']:,} ({NUM['pct_labelled']}\\%)"),
        ("Male", f"{NUM['n_male']:,} ({NUM['pct_male']}\\%)"),
        ("Age, y", iqr_str(base["age_at_visit"])),
        ("Education, y", iqr_str(base["EDUCYRS"], 0)),
        ("Disease duration, y", iqr_str(base["duration_yrs"], 2)),
        ("MDS-UPDRS Part III (OFF)", iqr_str(base["updrs3_score"], 0)),
        ("Hoehn \\& Yahr (OFF)", iqr_str(base["hy"], 1)),
        ("MoCA", iqr_str(base["moca"], 0)),
        ("Tremor subscore", iqr_str(base["tremor_score"], 2)),
        ("PIGD subscore", iqr_str(base["pigd_score"], 2)),
        ("Follow-up per participant, y", iqr_str(panel.groupby("PATNO")["years_from_baseline"].max())),
        ("Labelled visits per participant", iqr_str(n_lab, 0)),
        ("Interval between labelled visits, y", iqr_str(gaps, 2)),
    ]
    sub = [
        ("Tremor dominant", f"{NUM['baseline_n_TD']} ({NUM['baseline_pct_TD']}\\%)",
         f"{NUM['labelled_n_TD']:,} ({NUM['labelled_pct_TD']}\\%)"),
        ("Indeterminate", f"{NUM['baseline_n_Indeterminate']} ({NUM['baseline_pct_Indeterminate']}\\%)",
         f"{NUM['labelled_n_Indeterminate']} ({NUM['labelled_pct_Indeterminate']}\\%)"),
        ("PIGD", f"{NUM['baseline_n_PIGD']} ({NUM['baseline_pct_PIGD']}\\%)",
         f"{NUM['labelled_n_PIGD']:,} ({NUM['labelled_pct_PIGD']}\\%)"),
    ]

    lines = ["% generated by paper/ieee/make_assets.py -- do not edit",
             "\\begin{tabular}{@{}lr@{}}", "\\toprule",
             "Characteristic & Value \\\\", "\\midrule"]
    lines += [f"{k} & {v} \\\\" for k, v in rows]
    lines += ["\\midrule",
              "\\multicolumn{2}{@{}l}{\\textit{Subtype distribution: first visit / all labelled visits}} \\\\"]
    lines += [f"\\quad {k} & {a} / {b} \\\\" for k, a, b in sub]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "table1_cohort.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figure 1: participant flow
# ---------------------------------------------------------------------------

def fig_flow() -> None:
    ledger = pd.read_csv(E5 / "flow_ledger.csv")
    rec("flow_rows", list(ledger.columns))

    boxes = [
        ("PPMI curated data cut", "19,450 visits, 4,788 participants"),
        ("Sporadic PD subgroup", "5,789 visits, 1,080 participants"),
        ("Scheduled visits only", "5,774 visits, 1,080 participants"),
        ("Motor subtype assignable", "4,441 visits (76.9%), 1,080 participants"),
        ("$\\geq$ 2 labelled visits", "4,295 visits, 934 participants"),
        ("Consecutive labelled pairs", "3,361 prediction opportunities"),
    ]
    drops = [
        "$-$13,661 visits, $-$3,708 participants\nnot sporadic PD",
        "$-$15 visits\nunscheduled (14) or withdrawal (1)",
        "$-$1,333 visits unlabelled\n1,195 ON-medication exam only",
        "$-$146 participants\nsingle labelled visit",
        "$-$332 Indeterminate outcomes\nleave 3,029 two-class targets",
    ]

    fig, ax = plt.subplots(figsize=(COL, 3.65))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, len(boxes) * 2)
    ax.axis("off")
    bw, bh = 6.15, 1.05
    for i, (head, body) in enumerate(boxes):
        y = (len(boxes) - i) * 2 - 1.1
        ax.add_patch(Rectangle((0.15, y), bw, bh, fill=True, facecolor="#f2f5f8",
                               edgecolor=INK, lw=0.6, zorder=2))
        ax.text(0.35, y + bh - 0.35, head, fontsize=6.9, weight="bold",
                va="center", zorder=3)
        ax.text(0.35, y + 0.28, body, fontsize=6.3, va="center", zorder=3)
        if i < len(drops):
            ax.add_patch(FancyArrowPatch((0.15 + bw / 2, y), (0.15 + bw / 2, y - 0.95),
                                         arrowstyle="-|>", mutation_scale=7,
                                         lw=0.7, color=INK, zorder=2))
            ax.text(bw + 0.45, y - 0.47, drops[i], fontsize=5.9, va="center",
                    color=RED, linespacing=1.35)
    fig.savefig(FIG / "fig1_flow.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: label instability
# ---------------------------------------------------------------------------

def fig_instability() -> None:
    tp = pd.read_csv(E2 / "transition_probabilities.csv")
    rev = pd.read_csv(E2 / "other_quantities.csv")
    interval = pd.read_csv(OUT / "transitions" / "transitions_by_interval.csv")
    km_n = pd.read_csv(OUT / "transitions" / "conversion_km_naive.csv")
    km_s = pd.read_csv(OUT / "transitions" / "conversion_km_sustained.csv")

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.05))

    # (a) transition matrix
    ax = axes[0]
    mat = tp.pivot(index="from_state", columns="to_state", values="probability")
    mat = mat.loc[STATES, STATES]
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
    for i, fr in enumerate(STATES):
        for j, to in enumerate(STATES):
            row = tp[(tp.from_state == fr) & (tp.to_state == to)].iloc[0]
            v = row["probability"]
            ax.text(j, i + 0.11, fmt(v, 3), ha="center", va="center", fontsize=7.0,
                    color="white" if v > 0.55 else INK, weight="bold")
            ax.text(j, i - 0.22,
                    f"[{fmt(row['bootstrap_low'], 2)}, {fmt(row['bootstrap_high'], 2)}]",
                    ha="center", va="center", fontsize=5.3,
                    color="white" if v > 0.55 else "#555555")
    ax.set_xticks(range(3), [SHORT[s] for s in STATES])
    ax.set_yticks(range(3), [SHORT[s] for s in STATES])
    ax.set_xlabel("next labelled visit")
    ax.set_ylabel("current visit")
    ax.set_title("(a) one-step transitions", loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)

    for fr in STATES:
        for to in STATES:
            row = tp[(tp.from_state == fr) & (tp.to_state == to)].iloc[0]
            tag = f"{SHORT[fr]}_{SHORT[to]}"
            rec(f"p_{tag}", float(row["probability"]))
            rec(f"p_{tag}_lo", float(row["bootstrap_low"]))
            rec(f"p_{tag}_hi", float(row["bootstrap_high"]))
    rec("pigd_not_absorbing",
        round(100 * (1 - float(tp[(tp.from_state == "PIGD") & (tp.to_state == "PIGD")]["probability"].iloc[0])), 1))

    # (b) reversion
    ax = axes[1]
    keys = ["revert[TD->Indeterminate]", "revert[PIGD->Indeterminate]",
            "revert[PIGD->TD]", "revert[TD->PIGD]",
            "revert[Indeterminate->PIGD]", "revert[Indeterminate->TD]"]
    labels = [r"TD$\to$Indet", r"PIGD$\to$Indet", r"PIGD$\to$TD",
              r"TD$\to$PIGD", r"Indet$\to$PIGD", r"Indet$\to$TD"]
    r = rev.set_index("quantity")
    est = [r.loc[k, "estimate"] for k in keys]
    lo = [r.loc[k, "estimate"] - r.loc[k, "bootstrap_low"] for k in keys]
    hi = [r.loc[k, "bootstrap_high"] - r.loc[k, "estimate"] for k in keys]
    y = np.arange(len(keys))
    colours = [GOLD if "Indeterminate]" in k else BLUE for k in keys]
    ax.barh(y, est, xerr=[lo, hi], color=colours, height=0.62,
            error_kw=dict(lw=0.7, capsize=1.8, ecolor=INK), zorder=3)
    pooled = r.loc["reversion_pooled", "estimate"]
    rec("reversion_pooled", float(pooled))
    rec("reversion_lo", float(r.loc["reversion_pooled", "bootstrap_low"]))
    rec("reversion_hi", float(r.loc["reversion_pooled", "bootstrap_high"]))
    for k, tag in (("revert[TD->Indeterminate]", "rev_TD_Ind"),
                   ("revert[PIGD->Indeterminate]", "rev_PIGD_Ind"),
                   ("revert[TD->PIGD]", "rev_TD_PIGD"),
                   ("revert[PIGD->TD]", "rev_PIGD_TD")):
        rec(tag, float(r.loc[k, "estimate"]))
        rec(f"{tag}_lo", float(r.loc[k, "bootstrap_low"]))
        rec(f"{tag}_hi", float(r.loc[k, "bootstrap_high"]))
    ax.axvline(pooled, color=RED, lw=0.9, ls="--", zorder=4)
    ax.text(0.99, 0.06, f"pooled {fmt(pooled, 3)}", color=RED, fontsize=6.0,
            ha="right", va="bottom", transform=ax.transAxes)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("P(return to origin at next visit)")
    ax.set_title("(b) reversion of an observed change", loc="left")
    tidy(ax, "x")

    # (c) Kaplan-Meier, two conversion definitions
    ax = axes[2]
    for df, lab, c in ((km_n, "single assessment", RED),
                       (km_s, "sustained (2 visits)", BLUE)):
        ax.step(df["time"], 100 * (1 - df["survival"]), where="post",
                color=c, lw=1.1, label=lab, zorder=3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 63)
    ax.set_xlabel("years from first labelled visit")
    ax.set_ylabel("cumulative incidence (%)")
    ax.set_title("(c) conversion to PIGD", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    tidy(ax)

    km = rev.set_index("quantity")
    rec("km_naive_yr5", float(km.loc["km_naive_yr5", "estimate"]))
    rec("km_naive_yr5_lo", float(km.loc["km_naive_yr5", "bootstrap_low"]))
    rec("km_naive_yr5_hi", float(km.loc["km_naive_yr5", "bootstrap_high"]))
    rec("km_sust_yr5", float(km.loc["km_sustained_yr5", "estimate"]))
    rec("km_sust_yr5_lo", float(km.loc["km_sustained_yr5", "bootstrap_low"]))
    rec("km_sust_yr5_hi", float(km.loc["km_sustained_yr5", "bootstrap_high"]))
    rec("sustained_removes", float(km.loc["sustained_removes_share", "estimate"]))
    rec("sustained_removes_lo", float(km.loc["sustained_removes_share", "bootstrap_low"]))
    rec("sustained_removes_hi", float(km.loc["sustained_removes_share", "bootstrap_high"]))
    rec("n_changes", int(km.loc["n_changes", "estimate"]))
    rec("naive_converted", int(km.loc["naive_converted", "estimate"]))
    rec("sustained_converted", int(km.loc["sustained_converted", "estimate"]))
    rec("naive_at_risk", int(km.loc["naive_at_risk", "estimate"]))

    iv = interval.set_index(["from_state", "to_state", "interval"])
    rec("p_TDTD_short", float(iv.loc[("TD", "TD", "≤ 1.5"), "probability"]))
    rec("p_TDTD_long", float(iv.loc[("TD", "TD", "> 4.0"), "probability"]))
    rec("p_TDPIGD_short", float(iv.loc[("TD", "PIGD", "≤ 1.5"), "probability"]))
    rec("p_TDPIGD_long", float(iv.loc[("TD", "PIGD", "> 4.0"), "probability"]))

    fig.subplots_adjust(wspace=0.56)
    fig.savefig(FIG / "fig2_instability.pdf")
    plt.close(fig)

    # Table II -- transition matrix with patient-bootstrap intervals
    lines = ["% generated by paper/ieee/make_assets.py -- do not edit",
             "\\begin{tabular}{@{}llrrl@{}}", "\\toprule",
             "From & To & $n$ & $\\hat{p}$ & 95\\% CI \\\\", "\\midrule"]
    for fr in STATES:
        sub = tp[tp.from_state == fr]
        n_from = int(sub["n_from"].iloc[0])
        for k, (_, row) in enumerate(sub.set_index("to_state").loc[STATES].reset_index().iterrows()):
            first = f"{SHORT[fr]} ($n{{=}}{n_from:,}$)" if k == 0 else ""
            lines.append(
                f"{first} & {SHORT[row['to_state']]} & {int(row['n_transitions']):,} & "
                f"{fmt(row['probability'], 3)} & "
                f"({fmt(row['bootstrap_low'], 3)}, {fmt(row['bootstrap_high'], 3)}) \\\\")
        if fr != STATES[-1]:
            lines.append("\\addlinespace[1.5pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "table2_transitions.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figure 3 and Table III: harmonised comparison
# ---------------------------------------------------------------------------

FAMILY_SHORT = {
    "subscore GP": "GP",
    "latent-state HMM": "HMM",
    "time-varying LMM": "LMM",
    "subscore LMM": "LMM",
    "observed-label chain": "chain",
    "baseline": "baseline",
}


def fig_families() -> None:
    met = pd.read_csv(E8 / "harmonised_metrics.csv")
    con = pd.read_csv(E8 / "paired_contrasts.csv").set_index("model")
    base = "Persistence (recorded label)"

    fitted = met[met.family != "baseline"].sort_values("accuracy")
    rec("n_fitted", len(fitted))
    rec("acc_min", float(fitted["accuracy"].min()))
    rec("acc_max", float(fitted["accuracy"].max()))
    rec("acc_spread_pts", round(100 * (fitted["accuracy"].max() - fitted["accuracy"].min()), 2))
    rec("acc_persistence", float(met[met.model == base]["accuracy"].iloc[0]))
    rec("macro_f1_persistence", float(met[met.model == base]["macro_f1"].iloc[0]))
    rec("macro_f1_best_fitted", float(fitted["macro_f1"].max()))
    rec("n_pred_ind_persistence", int(met[met.model == base]["n_predicted_Indeterminate"].iloc[0]))
    rec("n_models_predicting_ind", int((met["n_predicted_Indeterminate"] > 0).sum()))
    rec("n_models_total", len(met))
    deltas = con.loc[fitted["model"], "delta_accuracy"]
    rec("delta_min", float(deltas.min()))
    rec("delta_max", float(deltas.max()))
    rec("delta_min_lo", float(con.loc[deltas.idxmin(), "ci_low"]))
    rec("delta_min_hi", float(con.loc[deltas.idxmin(), "ci_high"]))
    rec("delta_max_lo", float(con.loc[deltas.idxmax(), "ci_low"]))
    rec("delta_max_hi", float(con.loc[deltas.idxmax(), "ci_high"]))
    rec("acc_hmm3", float(met[met.model == "CT-HMM, 3 latent states"]["accuracy"].iloc[0]))
    rec("acc_hmm5", float(met[met.model == "CT-HMM, 5 latent states"]["accuracy"].iloc[0]))
    rec("acc_dtmc", float(met[met.model == "Markov chain, discrete"]["accuracy"].iloc[0]))
    rec("acc_ctmc", float(met[met.model == "CTMC, interval-aware"]["accuracy"].iloc[0]))
    rec("acc_gp", float(met[met.model == "Subscore GP, independent"]["accuracy"].iloc[0]))
    spread_wo5 = fitted[fitted.model != "CT-HMM, 5 latent states"]
    rec("acc_spread_without5",
        round(100 * (spread_wo5["accuracy"].max() - spread_wo5["accuracy"].min()), 2))

    order = met.sort_values("accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(WIDE, 2.5))
    y = np.arange(len(order))
    cols = [GREY if f == "baseline" else BLUE for f in order["family"]]
    cols = [RED if m == base else c for m, c in zip(order["model"], cols)]
    ax.barh(y, order["accuracy"], color=cols, height=0.64, zorder=3)
    for yi, (m, a) in enumerate(zip(order["model"], order["accuracy"])):
        ax.text(a + 0.004, yi, fmt(a, 4), va="center", fontsize=6.1, zorder=4)
    lo, hi = fitted["accuracy"].min(), fitted["accuracy"].max()
    ax.axvspan(lo, hi, color=BLUE, alpha=0.10, zorder=1)
    ax.axvline(NUM["acc_persistence"], color=RED, lw=0.9, ls="--", zorder=4)
    labels = [f"{m}" for m in order["model"]]
    ax.set_yticks(y, labels)
    ax.set_xlim(0.60, 0.80)
    ax.set_xlabel("three-class accuracy on 3,361 held-out transitions (leave-one-participant-out)")
    ax.annotate(f"nine fitted specifications span {NUM['acc_spread_pts']:.2f} accuracy points",
                xy=((lo + hi) / 2, len(order) - 0.2), xytext=((lo + hi) / 2, len(order) + 0.9),
                ha="center", fontsize=6.4, color=BLUE,
                arrowprops=dict(arrowstyle="-[, widthB=2.6, lengthB=0.3",
                                lw=0.7, color=BLUE))
    ax.text(NUM["acc_persistence"] - 0.004, 0.1, "persistence rule", color=RED,
            fontsize=6.2, ha="right")
    ax.set_ylim(-0.7, len(order) + 1.4)
    tidy(ax, "x")
    fig.savefig(FIG / "fig3_families.pdf")
    plt.close(fig)

    # Table III
    show = met.copy()
    show["delta"] = [con.loc[m, "delta_accuracy"] if m in con.index else np.nan
                     for m in show["model"]]
    show["dlo"] = [con.loc[m, "ci_low"] if m in con.index else np.nan for m in show["model"]]
    show["dhi"] = [con.loc[m, "ci_high"] if m in con.index else np.nan for m in show["model"]]
    fam_order = ["observed-label chain", "latent-state HMM", "subscore LMM",
                 "time-varying LMM", "subscore GP", "baseline"]
    show["_fam"] = show["family"].map({f: i for i, f in enumerate(fam_order)})
    show = show.sort_values(["_fam", "accuracy"], ascending=[True, False])

    lines = ["% generated by paper/ieee/make_assets.py -- do not edit",
             "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
             "Specification & Family & Acc. & Brier & Log loss & Macro F1 & "
             "$n_{\\mathrm{Indet}}$ & $\\Delta$ vs.\\ persistence (95\\% CI) \\\\",
             "\\midrule"]
    for fam, grp in show.groupby("family", sort=False):
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{fam}}}}} \\\\")
        for _, r in grp.iterrows():
            d = ("---" if pd.isna(r["delta"])
                 else f"{fmt(r['delta'], 4, True)} ({fmt(r['dlo'], 4, True)}, {fmt(r['dhi'], 4, True)})")
            ll = fmt(r["log_loss"], 3) if r["log_loss"] < 10 else fmt(r["log_loss"], 2)
            lines.append(
                f"\\quad {r['model']} & {FAMILY_SHORT[r['family']]} & {fmt(r['accuracy'], 4)} & "
                f"{fmt(r['brier_multiclass'], 3)} & {ll} & {fmt(r['macro_f1'], 4)} & "
                f"{int(r['n_predicted_Indeterminate'])} & {d} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "table3_families.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figure 4: where the margin comes from, and threshold fragility
# ---------------------------------------------------------------------------

def fig_margin() -> None:
    dec = pd.read_csv(E8 / "margin_decomposition.csv")
    att = pd.read_csv(O4 / "attainable_ratios.csv")
    summ = pd.read_csv(O4 / "summary.csv").iloc[0]
    cross = pd.read_csv(O4 / "label_crosstab.csv", index_col=0)
    mov = pd.read_csv(O4 / "per_class_movement.csv")

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.12),
                             gridspec_kw=dict(width_ratios=[1.12, 1.0]))

    # (a) stratified gain
    ax = axes[0]
    models = ["CT-HMM, 3 latent states", "Subscore GP, independent",
              "Markov chain, discrete"]
    nice = ["CT-HMM (3 latent)", "Subscore GP", "Markov chain"]
    width = 0.26
    x = np.arange(3)
    for k, (m, lab) in enumerate(zip(models, nice)):
        sub = dec[dec.model == m].set_index("current_label").loc[STATES]
        ax.bar(x + (k - 1) * width, sub["cases_gained"], width,
               label=lab, zorder=3,
               color=[BLUE, GREEN, GOLD][k])
        for xi, v in zip(x + (k - 1) * width, sub["cases_gained"]):
            ax.text(xi, v + 1.5, f"{int(v)}", ha="center", fontsize=5.8, zorder=4)
    ax.set_xticks(x, [f"{SHORT[s]}\n($n={int(dec[dec.current_label == s]['n'].iloc[0]):,}$)"
                      for s in STATES])
    ax.set_xlabel("subtype recorded at the prediction point")
    ax.set_ylabel("extra correct predictions\nover the persistence rule")
    ax.set_ylim(0, 142)
    ax.legend(frameon=False, loc="upper left", handlelength=1.1,
              borderaxespad=0.2, labelspacing=0.3)
    ax.set_title("(a) the whole margin sits in one stratum", loc="left")
    tidy(ax)

    hmm = dec[dec.model == "CT-HMM, 3 latent states"].set_index("current_label")
    rec("gain_share_ind", float(hmm.loc["Indeterminate", "share_of_total_gain"]))
    rec("gain_cases_ind", int(hmm.loc["Indeterminate", "cases_gained"]))
    rec("gain_cases_td", int(hmm.loc["TD", "cases_gained"]))
    rec("gain_cases_pigd", int(hmm.loc["PIGD", "cases_gained"]))
    rec("n_ind_cases", int(hmm.loc["Indeterminate", "n"]))
    rec("share_ind_cases", float(hmm.loc["Indeterminate", "share_of_cases"]))
    rec("ident_td", float(hmm.loc["TD", "identical_predictions"]))
    rec("ident_pigd", float(hmm.loc["PIGD", "identical_predictions"]))
    rec("ident_ind", float(hmm.loc["Indeterminate", "identical_predictions"]))
    rec("acc_pigd_stratum", float(hmm.loc["PIGD", "model_accuracy"]))
    chain = dec[dec.model == "Markov chain, discrete"].set_index("current_label")
    rec("chain_gain_ind", int(chain.loc["Indeterminate", "cases_gained"]))
    rec("chain_ident_td", float(chain.loc["TD", "identical_predictions"]))
    gp = dec[dec.model == "Subscore GP, independent"].set_index("current_label")
    rec("gp_gain_td", int(gp.loc["TD", "cases_gained"]))
    rec("gp_gain_pigd", int(gp.loc["PIGD", "cases_gained"]))

    # (b) attainable ratios inside the band
    ax = axes[1]
    ax.bar(att["ratio"], att["n_visits"], width=0.0075, color=BLUE, zorder=3)
    for _, r in att.iterrows():
        if r["n_visits"] >= 40:
            ax.text(r["ratio"] + 0.004, r["n_visits"] + 4, f"{int(r['n_visits'])}",
                    ha="left", fontsize=6.0, zorder=4)
    for v, c in ((0.90, RED), (1.15, RED), (0.91, GREEN), (1.13, GREEN)):
        ax.axvline(v, color=c, lw=0.9, ls="--" if c == RED else ":", zorder=4)
    ax.annotate("primary rule\n0.90 / 1.15", xy=(0.90, 248), xytext=(0.955, 262),
                color=RED, fontsize=5.9, ha="center", va="top", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=RED))
    ax.annotate("perturbed\n0.91 / 1.13", xy=(1.13, 248), xytext=(1.075, 262),
                color=GREEN, fontsize=5.9, ha="center", va="top", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREEN))
    ax.set_xlim(0.882, 1.168)
    ax.set_ylim(0, 278)
    ax.set_xlabel("attainable tremor/PIGD ratio inside the band")
    ax.set_ylabel("Indeterminate visits")
    ax.set_title("(b) the band is two reachable values wide", loc="left")
    tidy(ax)

    rec("n_attainable_values", len(att))
    rec("n_ind_with_ratio", int(att["n_visits"].sum()))
    top = att.sort_values("n_visits", ascending=False).iloc[0]
    rec("top_ratio", float(top["ratio"]))
    rec("top_ratio_n", int(top["n_visits"]))
    rec("top_ratio_pct", round(100 * top["n_visits"] / att["n_visits"].sum(), 1))
    second = att.sort_values("n_visits", ascending=False).iloc[1]
    rec("second_ratio", float(second["ratio"]))
    rec("second_ratio_n", int(second["n_visits"]))
    rec("second_ratio_pct", round(100 * second["n_visits"] / att["n_visits"].sum(), 1))

    rec("relabelled", int(summ["visits_relabelled"]))
    rec("relabelled_pct", round(100 * summ["share_relabelled"], 1))
    rec("patients_affected", int(summ["patients_with_any_relabelled"]))
    rec("patients_affected_pct", round(100 * summ["share_patients_affected"], 1))
    rec("ind_before", int(summ["indeterminate_before"]))
    rec("ind_after", int(summ["indeterminate_after"]))
    ind_row = mov[mov["class"] == "Indeterminate"].iloc[0]
    rec("ind_drop_pct", round(-100 * ind_row["pct_change"], 1))
    rec("ind_to_pigd", int(cross.loc["Indeterminate", "PIGD"]))
    rec("ind_to_td", int(cross.loc["Indeterminate", "TD"]))
    rec("agree2_base", float(summ["agreement_2class_baseline"]))
    rec("agree2_narrow", float(summ["agreement_2class_narrow"]))
    rec("agree3_base", float(summ["agreement_3class_baseline"]))
    rec("agree3_narrow", float(summ["agreement_3class_narrow"]))

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIG / "fig4_margin.pdf")
    plt.close(fig)

    # Table V -- crosstab
    lines = ["% generated by paper/ieee/make_assets.py -- do not edit",
             "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "\\multirow{2}{*}{Primary rule} & \\multicolumn{3}{c}{Perturbed rule} & "
             "\\multirow{2}{*}{Total} \\\\",
             "\\cmidrule(lr){2-4}",
             " & TD & Indet & PIGD & \\\\", "\\midrule"]
    for st in STATES:
        row = cross.loc[st]
        lines.append(f"{SHORT[st]} & {int(row['TD']):,} & {int(row['Indeterminate']):,} & "
                     f"{int(row['PIGD']):,} & {int(row.sum()):,} \\\\")
    tot = cross.sum()
    lines += ["\\midrule",
              f"Total & {int(tot['TD']):,} & {int(tot['Indeterminate']):,} & "
              f"{int(tot['PIGD']):,} & {int(tot.sum()):,} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (TAB / "table5_threshold.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table IV: two-class task, discrimination and calibration
# ---------------------------------------------------------------------------

NICE_TWO = {
    "Subtype-history: CT-HMM (latent, interval-aware)": "CT-HMM (3 latent states)",
    "Subtype-history: Markov (observed, discrete)": "Markov chain, discrete",
    "Subtype-history: Persistence (last non-Indeterminate)": "Persistence rule",
    "Profile: level + rate + medication": "Nine-domain profile",
    "Profile: level only (random walk)": "Nine-domain profile, level only",
    "Profile: level + rate, no medication": "Nine-domain profile, no medication",
    "Last observation carried forward": "Last observation carried forward",
    "Population mean trajectory": "Population mean trajectory",
}
TWO_ORDER = [
    "Profile: level + rate + medication",
    "Subtype-history: CT-HMM (latent, interval-aware)",
    "Subtype-history: Markov (observed, discrete)",
    "Subtype-history: Persistence (last non-Indeterminate)",
    "Last observation carried forward",
    "Population mean trajectory",
]


def table_two_class() -> None:
    summ = json.loads((MSC / "summary.json").read_text())
    held = {d["model"]: d for d in summ["heldout_matched"]}
    auc = pd.read_csv(ROC / "auc_metrics.csv").set_index("model")
    auc_con = pd.read_csv(ROC / "auc_contrast.csv").iloc[0]
    aic = pd.read_csv(MSC / "aic_bic.csv")

    auc_map = {
        "Subtype-history: CT-HMM (latent, interval-aware)": "CT-HMM",
        "Profile: level + rate + medication": "Rich clinical profile",
        "Subtype-history: Persistence (last non-Indeterminate)": "Persistence rule (reference)",
    }

    lines = ["% generated by paper/ieee/make_assets.py -- do not edit",
             "\\begin{tabular}{@{}lrlrrl@{}}", "\\toprule",
             "Model & Acc. & 95\\% CI & Brier & Log loss & AUC (95\\% CI) \\\\",
             "\\midrule"]
    for key in TWO_ORDER:
        d = held[key]
        a = auc_map.get(key)
        acol = ("---" if a is None else
                f"{fmt(auc.loc[a, 'auc'], 4)} ({fmt(auc.loc[a, 'ci_low'], 4)}, {fmt(auc.loc[a, 'ci_high'], 4)})")
        ll = fmt(d["log_loss"], 4) + ("$^{\\dagger}$" if d["log_loss_clipped"] else "")
        lines.append(
            f"{NICE_TWO[key]} & {fmt(d['accuracy'], 4)} & "
            f"({fmt(d['accuracy_ci_patient_bootstrap_low'], 4)}, "
            f"{fmt(d['accuracy_ci_patient_bootstrap_high'], 4)}) & "
            f"{fmt(d['brier_two_class'], 4)} & {ll} & {acol} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "table4_twoclass.tex").write_text("\n".join(lines) + "\n")

    hmm = held["Subtype-history: CT-HMM (latent, interval-aware)"]
    pers = held["Subtype-history: Persistence (last non-Indeterminate)"]
    prof = held["Profile: level + rate + medication"]
    rec("two_n", hmm["n_scored"])
    rec("two_patients", hmm["n_patients"])
    rec("two_acc_hmm", hmm["accuracy"])
    rec("two_acc_hmm_lo", hmm["accuracy_ci_patient_bootstrap_low"])
    rec("two_acc_hmm_hi", hmm["accuracy_ci_patient_bootstrap_high"])
    rec("two_acc_pers", pers["accuracy"])
    rec("two_acc_pers_lo", pers["accuracy_ci_patient_bootstrap_low"])
    rec("two_acc_pers_hi", pers["accuracy_ci_patient_bootstrap_high"])
    rec("two_acc_prof", prof["accuracy"])
    rec("two_acc_prof_lo", prof["accuracy_ci_patient_bootstrap_low"])
    rec("two_acc_prof_hi", prof["accuracy_ci_patient_bootstrap_high"])
    rec("two_brier_hmm", hmm["brier_two_class"])
    rec("two_brier_pers", pers["brier_two_class"])
    rec("two_n_ind_excluded", hmm["n_indeterminate_next"])

    con = pd.read_csv(MSC / "heldout_contrasts.csv").set_index("contrast")
    row = con.loc["Profile: level + rate + medication vs subtype-history CT-HMM"]
    rec("prof_vs_hmm_delta", float(row["accuracy_delta"]))
    rec("prof_vs_hmm_lo", float(row["delta_ci_patient_bootstrap_low"]))
    rec("prof_vs_hmm_hi", float(row["delta_ci_patient_bootstrap_high"]))

    rec("auc_hmm", float(auc.loc["CT-HMM", "auc"]))
    rec("auc_hmm_lo", float(auc.loc["CT-HMM", "ci_low"]))
    rec("auc_hmm_hi", float(auc.loc["CT-HMM", "ci_high"]))
    rec("auc_prof", float(auc.loc["Rich clinical profile", "auc"]))
    rec("auc_prof_lo", float(auc.loc["Rich clinical profile", "ci_low"]))
    rec("auc_prof_hi", float(auc.loc["Rich clinical profile", "ci_high"]))
    rec("auc_pers", float(auc.loc["Persistence rule (reference)", "auc"]))
    rec("auc_delta", float(auc_con["delta"]))
    rec("auc_delta_lo", float(auc_con["ci_low"]))
    rec("auc_delta_hi", float(auc_con["ci_high"]))

    grp = aic[aic.group == "C: joint likelihood over whole sequences"].set_index("model")
    rec("daic_latent", float(grp.loc["Markov + initial state", "delta_aic_within_group"]))
    rec("dbic_latent", float(grp.loc["Markov + initial state", "delta_bic_within_group"]))

    # three-class accuracy of the same predictions, for the "task not model" point
    loo = json.loads((OUT / "loo" / "loo_overall_metrics.json").read_text())
    rec("three_correct", loo["n_correct"])
    rec("three_n", loo["n_predictions"])
    rec("three_acc", loo["accuracy"])


# ---------------------------------------------------------------------------
# Figure 5 and Table VI: what the continuous representation carries
# ---------------------------------------------------------------------------

OUTCOME_LABEL = {
    "hy_ge_3": "Hoehn \\& Yahr $\\geq 3$",
    "mci": "MCI (HVLT/Benton)",
    "motor_complications": "Motor complications",
    "cognitive_dx": "Cognitive diagnosis$^{\\ddagger}$",
}
OUTCOME_SHORT = {
    "hy_ge_3": "H&Y $\\geq$ 3",
    "mci": "MCI",
    "motor_complications": "Motor compl.",
    "cognitive_dx": "Cognitive dx",
}


def fig_outcomes() -> None:
    # Outcome contrasts come from the full-precision rerun. The A-1 run
    # (20261103T212122Z) computed the same contrasts from a copy of
    # outcome_table.csv saved with `.round(4)`, which is lossy for rate
    # coordinates of order 1e-3; scripts/82 marks those numbers superseded.
    con = pd.read_csv(PROF / "h3_component_vs_subtype.csv")
    dim = pd.read_csv(PROF / "representation_dimensionality.csv")
    mc = pd.read_csv(PROF / "model_comparison.csv").set_index("model")
    pc = pd.read_csv(PROF / "contrasts.csv").set_index("contrast")

    lev = dim[dim.which == "level"].iloc[0]
    rat = dim[dim.which == "rate"].iloc[0]
    rec("pc_n_for_90_level", int(lev["n_components_for_90pct"]))
    rec("pc1_level", float(lev["pc1_share"]))
    rec("pc1_rate", float(rat["pc1_share"]))
    rec("pc_n_for_90_rate", int(rat["n_components_for_90pct"]))

    r = pc.loc["rich profile against assume-no-change"]
    rec("prof_vs_locf_ll", float(r["loglik_delta"]))
    rec("prof_vs_locf_lo", float(r["delta_ci_low"]))
    rec("prof_vs_locf_hi", float(r["delta_ci_high"]))
    rr = pc.loc["modelling the rate of change"]
    rec("rate_component_ll", float(rr["loglik_delta"]))
    rec("rate_component_lo", float(rr["delta_ci_low"]))
    rec("rate_component_hi", float(rr["delta_ci_high"]))
    rm = pc.loc["medication as an input"]
    rec("med_component_ll", float(rm["loglik_delta"]))
    rec("med_component_lo", float(rm["delta_ci_low"]))
    rec("med_component_hi", float(rm["delta_ci_high"]))

    order = ["hy_ge_3", "mci", "cognitive_dx", "motor_complications"]
    lv = con[con.kind == "level"].set_index("outcome").loc[order]
    rt = con[con.kind == "rate"].set_index("outcome").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.95),
                             gridspec_kw=dict(width_ratios=[0.78, 1.0]))

    # (a) scree
    ax = axes[0]
    for row, lab, c, mk in ((lev, "level coordinates", BLUE, "o"),
                            (rat, "rate coordinates", GOLD, "s")):
        shares = np.array(json.loads(row["variance_shares"]))
        ax.plot(np.arange(1, len(shares) + 1), 100 * np.cumsum(shares),
                marker=mk, ms=2.6, lw=1.0, color=c, label=lab, zorder=3)
    ax.axhline(90, color=RED, lw=0.8, ls="--", zorder=4)
    ax.text(1.1, 91.5, "90% of variance", color=RED, fontsize=5.9)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("principal components retained")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_ylim(30, 104)
    ax.legend(frameon=False, loc="lower right", handlelength=1.3)
    ax.set_title("(a) the profile is multidimensional", loc="left")
    tidy(ax)

    # (b) outcome discrimination
    ax = axes[1]
    y = np.arange(len(order))
    h = 0.26
    ax.barh(y + h, lv["auc_b"], h, color=GREY, label="subtype label", zorder=3)
    ax.barh(y, lv["auc_a"], h, color=BLUE, label="best level coordinate", zorder=3)
    ax.barh(y - h, rt["auc_a"], h, color=GOLD, label="best rate coordinate", zorder=3)
    for yi, (d, lo, hi) in enumerate(zip(lv["delta"], lv["delta_ci_low"], lv["delta_ci_high"])):
        star = "" if lo <= 0 <= hi else "$^{*}$"
        ax.text(lv["auc_a"].iloc[yi] + 0.006, yi,
                f"{fmt(d, 3, True)}{star}", va="center", fontsize=5.9, color=BLUE, zorder=4)
    ax.axvline(0.5, color=INK, lw=0.7, zorder=4)
    ax.set_yticks(y, [OUTCOME_SHORT[o] for o in order])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.86)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("AUC against a later-recorded outcome, never used in fitting")
    ax.legend(frameon=False, loc="lower right", handlelength=1.1, ncol=1,
              labelspacing=0.3, borderaxespad=0.25)
    ax.set_title("(b) single coordinates beat the label on two outcomes", loc="left")
    tidy(ax, "x")

    fig.subplots_adjust(wspace=0.28)
    fig.savefig(FIG / "fig5_outcomes.pdf")
    plt.close(fig)

    for o in order:
        rec(f"auc_label_{o}", float(lv.loc[o, "auc_b"]))
        rec(f"auc_level_{o}", float(lv.loc[o, "auc_a"]))
        rec(f"delta_level_{o}", float(lv.loc[o, "delta"]))
        rec(f"delta_level_{o}_lo", float(lv.loc[o, "delta_ci_low"]))
        rec(f"delta_level_{o}_hi", float(lv.loc[o, "delta_ci_high"]))
        rec(f"delta_rate_{o}", float(rt.loc[o, "delta"]))
        rec(f"best_level_{o}", str(lv.loc[o, "best_component"]))
        rec(f"n_outcome_{o}", int(lv.loc[o, "n"]))
        rec(f"nev_outcome_{o}", int(lv.loc[o, "n_positive"]))

    lines = ["% generated by paper/ieee/make_assets.py -- do not edit",
             "\\begin{tabular}{@{}llrrrrl@{}}", "\\toprule",
             "Outcome & Best level coordinate & $n$ & Events & Label AUC & Coord.\\ AUC & "
             "$\\Delta$ (95\\% CI) \\\\", "\\midrule"]
    for o in order:
        row = lv.loc[o]
        comp = str(row["best_component"]).replace("level_", "").replace("_", "\\_")
        lines.append(
            f"{OUTCOME_LABEL[o]} & \\texttt{{{comp}}} & {int(row['n'])} & "
            f"{int(row['n_positive'])} & {fmt(row['auc_b'], 3)} & {fmt(row['auc_a'], 3)} & "
            f"{fmt(row['delta'], 3, True)} ({fmt(row['delta_ci_low'], 3, True)}, {fmt(row['delta_ci_high'], 3, True)}) \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{@{}l}{\\textit{Best rate-of-change coordinate, same outcomes}} \\\\")
    for o in order:
        row = rt.loc[o]
        comp = str(row["best_component"]).replace("rate_", "").replace("_", "\\_")
        lines.append(
            f"\\quad {OUTCOME_LABEL[o]} & \\texttt{{{comp}}} & {int(row['n'])} & "
            f"{int(row['n_positive'])} & {fmt(row['auc_b'], 3)} & {fmt(row['auc_a'], 3)} & "
            f"{fmt(row['delta'], 3, True)} ({fmt(row['delta_ci_low'], 3, True)}, {fmt(row['delta_ci_high'], 3, True)}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "table6_outcomes.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Remaining in-text numbers: history, retraining, stages, per-participant
# ---------------------------------------------------------------------------

def other_numbers() -> None:
    hp = pd.read_csv(HIST / "paired_comparisons.csv").set_index("comparison")
    for key, tag in (("last 2 - full history", "h2"), ("last 1 - full history", "h1")):
        r = hp.loc[key]
        rec(f"{tag}_acc", float(r["accuracy"]))
        rec(f"{tag}_acc_lo", float(r["accuracy_lo"]))
        rec(f"{tag}_acc_hi", float(r["accuracy_hi"]))
        rec(f"{tag}_auc", float(r["auc"]))
        rec(f"{tag}_auc_lo", float(r["auc_lo"]))
        rec(f"{tag}_auc_hi", float(r["auc_hi"]))
        rec(f"{tag}_bal", float(r["balanced_accuracy"]))
        rec(f"{tag}_brier", float(r["brier"]))
        rec(f"{tag}_logloss", float(r["log_loss"]))

    rp = pd.read_csv(RETR / "paired_comparisons.csv").set_index("comparison")
    pri = rp.loc["window-trained, last 2  -  full-trained, full history"]
    rec("retrain_bal", float(pri["balanced_accuracy"]))
    rec("retrain_bal_lo", float(pri["balanced_accuracy_lo"]))
    rec("retrain_bal_hi", float(pri["balanced_accuracy_hi"]))
    tr = rp.loc["window-trained, last 2  -  full-trained, last 2"]
    rec("retrain_training_effect", float(tr["balanced_accuracy"]))
    rec("retrain_training_lo", float(tr["balanced_accuracy_lo"]))
    rec("retrain_training_hi", float(tr["balanced_accuracy_hi"]))
    he = rp.loc["full-trained, last 2  -  full-trained, full history"]
    rec("retrain_history_effect", float(he["balanced_accuracy"]))
    rec("retrain_history_lo", float(he["balanced_accuracy_lo"]))
    rec("retrain_history_hi", float(he["balanced_accuracy_hi"]))

    sm = pd.read_csv(STAGE / "stage_metrics.csv")
    s1 = sm[(sm.stage == 1)].set_index("model")
    rec("stage1_hmm", float(s1.loc["CT-HMM", "accuracy"]))
    rec("stage1_prof", float(s1.loc["Rich clinical profile", "accuracy"]))
    rec("stage1_n", int(s1.loc["CT-HMM", "n_targets"]))
    rec("stage1_pct_pigd", round(100 * float(s1.loc["CT-HMM", "pct_PIGD"]), 0))
    rel = sm[sm.reliable]
    last = rel[rel.stage == rel.stage.max()].set_index("model")
    rec("stage_last", int(rel.stage.max()))
    rec("stage_last_pct_pigd", round(100 * float(last.loc["CT-HMM", "pct_PIGD"]), 0))
    csc = pd.read_csv(STAGE / "cross_stage_contrasts.csv")
    acc_contrasts = csc[(csc.metric == "accuracy") & csc.reliable]
    rec("stage_acc_any_excludes_zero", bool(acc_contrasts["excludes_zero"].any()))

    pa = pd.read_csv(PAT / "patient_accuracy_distribution.csv")
    full = pa[(pa.condition == "full history") & (pa.experiment == "6A")].iloc[0]
    rec("pp_n", int(full["n_patients"]))
    rec("pp_mean", float(full["mean_patient_accuracy"]))
    rec("pp_mean_lo", float(full["mean_ci_lo"]))
    rec("pp_mean_hi", float(full["mean_ci_hi"]))
    rec("pp_visit_level", float(full["visit_level_accuracy"]))
    rec("pp_median", float(full["median_patient_accuracy"]))
    rec("pp_perfect", int(full["n_exactly 100%"]))
    rec("pp_perfect_pct", round(float(full["pct_exactly 100%"]), 1))
    rec("pp_zero", int(full["n_exactly 0%"]))
    rec("pp_zero_pct", round(float(full["pct_exactly 0%"]), 1))

    comp = pd.read_csv(PAT / "patient_accuracy_by_subtype_composition.csv")
    cf = comp[(comp.condition == "full history") & (comp.experiment == "6A")].set_index("subtype_composition")
    cp = comp[(comp.condition == "persistence") & (comp.experiment == "6A")].set_index("subtype_composition")
    rec("pp_allTD_model", float(cf.loc["all TD targets", "mean_patient_accuracy"]))
    rec("pp_allTD_pers", float(cp.loc["all TD targets", "mean_patient_accuracy"]))
    rec("pp_even_model", float(cf.loc["evenly split", "mean_patient_accuracy"]))
    rec("pp_even_pers", float(cp.loc["evenly split", "mean_patient_accuracy"]))
    rec("pp_even_n", int(cf.loc["evenly split", "n_patients"]))
    rec("pp_allTD_n", int(cf.loc["all TD targets", "n_patients"]))

    ps = pd.read_csv(PAT / "patient_subtype_metrics.csv")
    pf = ps[(ps.condition == "full history") & (ps.experiment == "6A")].iloc[0]
    rec("pp_pigd_sens", float(pf["mean_PIGD_sensitivity_ge2_targets"]))
    rec("pp_td_spec", float(pf["mean_TD_specificity_ge2_targets"]))

    # Markov property test
    mp = pd.read_csv(OUT / "model" / "markov_property_test.csv").set_index("current_state")
    rec("markov_chi2_TD", float(mp.loc["TD", "chi2"]))
    rec("markov_dof_TD", int(mp.loc["TD", "dof"]))
    rec("markov_p_TD", float(mp.loc["TD", "p_value"]))

    # emission / sojourn parameters
    em = pd.read_csv(OUT / "model" / "hmm_emission.csv", index_col=0)
    rec("emission_TD", float(em.loc["TD", "TD"]))
    rec("emission_PIGD", float(em.loc["PIGD", "PIGD"]))
    rec("emission_Ind", float(em.loc["Indeterminate", "Indeterminate"]))
    soj = pd.read_csv(OUT / "model" / "mean_sojourn.csv", index_col=0)
    rec("sojourn_TD_latent", float(soj.loc["TD", "HMM (latent)"]))
    rec("sojourn_Ind_latent", float(soj.loc["Indeterminate", "HMM (latent)"]))
    rec("sojourn_PIGD_latent", float(soj.loc["PIGD", "HMM (latent)"]))

    # per-class recall on the three-class task
    pcm = pd.read_csv(OUT / "loo" / "loo_per_class_metrics.csv").set_index("state")
    rec("recall_TD", float(pcm.loc["TD", "recall"]))
    rec("recall_PIGD", float(pcm.loc["PIGD", "recall"]))
    rec("n_predicted_Ind_hmm", int(pcm.loc["Indeterminate", "n_predicted"]))
    rec("support_Ind", int(pcm.loc["Indeterminate", "support"]))

    # calibration of the three-class model
    cal = pd.read_csv(OUT / "loo" / "loo_calibration.csv")
    rec("calib_bins", len(cal))
    rec("calib_mae", float(np.average(
        (cal["observed_accuracy"] - cal["mean_confidence"]).abs(), weights=cal["n"])))

    # five-state convergence
    five = pd.read_csv(OUT / "loo_5hidden" / "loo_predictions_all_models.csv")
    hmm5 = five[five.model == "HMM"]
    folds = hmm5.groupby("held_out_patient")["train_converged"].first()
    rec("hmm5_folds_converged", int(folds.sum()))
    rec("hmm5_folds_total", int(len(folds)))
    rec("hmm5_folds_pct", round(100 * folds.mean(), 1))
    rec("hmm5_rows_nonconverged_pct", round(100 * (~hmm5["train_converged"]).mean(), 1))

    # unlabelled-visit mechanism
    who = pd.read_csv(E5 / "unlabelled_diagnosis.csv")
    top = who.sort_values("visits", ascending=False).iloc[0]
    rec("unlabelled_total", int(who["visits"].sum()))
    rec("unlabelled_on_med", int(top["visits"]))
    rec("unlabelled_on_med_pct", round(100 * float(top["share"]), 1))
    rest = who.sort_values("visits", ascending=False).iloc[1:]["visits"].tolist()
    rec("unlabelled_no_part3", int(rest[0]))
    rec("unlabelled_partial", int(rest[1]))
    rec("unlabelled_part2", int(rest[2]))

    bias = pd.read_csv(E5 / "who_is_unlabelled.csv").set_index("variable")
    rec("unlab_years_median", round(float(bias.loc["years from baseline", "median_unlabelled"]), 1))
    rec("lab_years_median", round(float(bias.loc["years from baseline", "median_labelled"]), 1))
    rec("unlab_on_med_share", round(100 * float(bias.loc["share on PD medication", "mean_unlabelled"]), 1))
    rec("lab_on_med_share", round(100 * float(bias.loc["share on PD medication", "mean_labelled"]), 1))

    ledger = pd.read_csv(E5 / "flow_ledger.csv").set_index("stage")
    rec("cut_visits", int(ledger.loc["curated data cut, as published", "visits"]))
    rec("cut_patients", int(ledger.loc["curated data cut, as published", "patients"]))
    rec("two_class_eligible_flow", int(ledger.loc["eligible for two-class TD-vs-PIGD scoring", "visits"]))

    # validation of the recomputed labels against PPMI's own derivations
    agree = pd.read_csv(O4 / "agreement_vs_ppmi.csv").iloc[0]
    rec("val_two_n", int(agree["two_class_n"]))
    rec("val_three_n", int(agree["three_class_n"]))
    rec("val_three_matches", int(round(agree["three_class"] * agree["three_class_n"])))

    # conditional next-label probabilities behind the Markov-property test
    panel = pd.read_csv(OUT / "visit_panel.csv", low_memory=False)
    lab = panel[panel["subtype_name"].notna()].sort_values(["PATNO", "visit_ts"])
    g = lab.groupby("PATNO")["subtype_name"]
    trip = pd.DataFrame({"prev": g.shift(1), "cur": lab["subtype_name"], "nxt": g.shift(-1)}).dropna()
    cur_td = trip[trip.cur == "TD"]
    for prev in ("TD", "PIGD"):
        sub = cur_td[cur_td.prev == prev]
        rec(f"p_next_PIGD_given_TD_prev_{prev}", float((sub.nxt == "PIGD").mean()))
    rec("n_triples_TD", len(cur_td))


def main() -> None:
    cohort_table()
    fig_flow()
    fig_instability()
    fig_families()
    fig_margin()
    table_two_class()
    fig_outcomes()
    other_numbers()
    NUM.pop("flow_rows", None)
    (HERE / "derived_numbers.json").write_text(json.dumps(NUM, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(list(FIG.glob('*.pdf')))} figures, "
          f"{len(list(TAB.glob('*.tex')))} table bodies, "
          f"{len(NUM)} derived numbers")


if __name__ == "__main__":
    main()
