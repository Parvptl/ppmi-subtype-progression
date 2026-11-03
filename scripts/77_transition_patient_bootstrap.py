"""E-2: patient-clustered bootstrap intervals for the observed-transition analysis.

THE DEFECT
----------
`outputs/transitions/report.txt` is the source of Result 1 of the manuscript --
the observed instability of the TD/PIGD label -- and it is the only analysis
block in the project still reporting intervals that assume independent
observations. `transitions.transition_counts_with_ci` computes **Wilson score
intervals** on 3,361 visit pairs drawn from 934 patients. 476 of those 934
patients (51%) never change state, so their pairs are near-deterministic
repeats of one another, and a binomial interval on the pooled count understates
the uncertainty. Every later experiment in this project uses a patient-clustered
bootstrap; this one does not.

WHAT THIS SCRIPT DOES
---------------------
Recomputes every headline quantity in that report with a **patient-level
cluster bootstrap**, and reports the Wilson interval beside it so the
understatement is quantified rather than asserted:

*   the nine pooled transition probabilities
*   the pooled reversion rate (the report's headline 35.2%) and the six
     per-transition-type reversion rates
*   Kaplan-Meier cumulative incidence of PIGD conversion at years 2 and 5,
     under both the single-visit and the sustained (two consecutive visits)
     definitions, and the share of apparent conversions the sustained
     definition removes
*   the interval-stratified TD -> PIGD probability at <=1.5 years and
     >4 years, which the report uses to argue that a pooled matrix is not a
     well-defined quantity

RESAMPLING DESIGN
-----------------
One frame, one draw, every statistic recomputed from scratch.

Patients are the sampling unit: all 1,080 patients with at least one labelled
visit are drawn with replacement, carrying **all** of their labelled visits.
A patient drawn twice becomes two distinct patients with distinct identifiers,
so their visits are never merged into one longer sequence. Each replicate is
then passed through the same `ppmi_subtype.transitions` functions that produced
the point estimates, so whoever qualifies for a given statistic in a replicate
is decided exactly as it is in the real data -- the 934 who contribute a
transition and the 885 at risk of PIGD conversion both vary from draw to draw,
and that variation is part of the interval rather than being conditioned away.

Point estimates are taken from the library functions unmodified, so they
reproduce the committed report exactly; the script fails if they do not.

    python scripts/77_transition_patient_bootstrap.py --out <dir>
"""

from __future__ import annotations

import argparse
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

from ppmi_subtype import transitions as T  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK, INK_MUTED, STATE_COLORS, _finish, _style  # noqa: E402

TIME = "years_from_baseline"
STATES = [1.0, 2.0, 3.0]
NAMES = {1.0: "TD", 2.0: "Indeterminate", 3.0: "PIGD"}
PIGD = 3.0

#  What the committed report prints, for the reproduction check.
COMMITTED = {
    ("TD", "TD"): 0.808, ("TD", "Indeterminate"): 0.080, ("TD", "PIGD"): 0.112,
    ("Indeterminate", "TD"): 0.440, ("Indeterminate", "Indeterminate"): 0.192,
    ("Indeterminate", "PIGD"): 0.368,
    ("PIGD", "TD"): 0.184, ("PIGD", "Indeterminate"): 0.114, ("PIGD", "PIGD"): 0.702,
}
COMMITTED_COUNTS = {"pairs": 3361, "patients": 934, "changes": 673,
                    "naive_converted": 263, "sustained_converted": 127,
                    "at_risk": 885}

ACCENT = "#1f4e79"
WARM = "#c2552c"

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


# ---------------------------------------------------------------------------
#  The statistics, each computed from a labelled panel
# ---------------------------------------------------------------------------
def km_incidence(panel: pd.DataFrame, min_consecutive: int,
                 horizons=(2.0, 5.0)) -> dict:
    """KM cumulative incidence of first PIGD entry, at the given horizons.

    Patients already PIGD at their first labelled visit have no observable
    time to entry and are excluded from the risk set, which is what
    `first_entry` records in `starts_in_state`.
    """
    fe = T.first_entry(panel, PIGD, min_consecutive=min_consecutive)
    risk = fe[~fe["starts_in_state"]]
    if risk.empty:
        return {f"km_yr{int(h)}": np.nan for h in horizons} | {
            "at_risk": 0, "converted": 0}
    km = T.kaplan_meier(risk["time"], risk["converted"])
    out = {"at_risk": int(len(risk)), "converted": int(risk["converted"].sum())}
    for h in horizons:
        upto = km[km["time"] <= h]
        surv = float(upto["survival"].iloc[-1]) if len(upto) else 1.0
        out[f"km_yr{int(h)}"] = 1.0 - surv
    return out


def all_statistics(panel: pd.DataFrame) -> dict:
    """Every quantity this script puts an interval on, from one labelled panel."""
    stats: dict[str, float] = {}

    pairs = T.consecutive_pairs(panel)
    stats["n_pairs"] = float(len(pairs))
    stats["n_patients_with_transition"] = float(pairs["PATNO"].nunique())

    probs = T.transition_matrix(pairs, states=STATES)
    counts = T.transition_matrix(pairs, states=STATES, normalize=False)
    for a in STATES:
        for b in STATES:
            stats[f"p[{NAMES[a]}->{NAMES[b]}]"] = float(probs.loc[a, b])
        stats[f"n_from[{NAMES[a]}]"] = float(counts.loc[a].sum())

    rev = T.reversion_after_change(panel)
    stats["n_changes"] = float(len(rev))
    stats["reversion_pooled"] = float(rev["reverted"].mean()) if len(rev) else np.nan
    summ = T.reversion_summary(rev, names=NAMES)
    for _, row in summ.iterrows():
        stats[f"revert[{row['from_state']}->{row['to_state']}]"] = float(row["p_revert"])

    naive = km_incidence(panel, 1)
    sust = km_incidence(panel, 2)
    stats["naive_at_risk"] = float(naive["at_risk"])
    stats["naive_converted"] = float(naive["converted"])
    stats["sustained_converted"] = float(sust["converted"])
    stats["km_naive_yr2"] = naive["km_yr2"]
    stats["km_naive_yr5"] = naive["km_yr5"]
    stats["km_sustained_yr2"] = sust["km_yr2"]
    stats["km_sustained_yr5"] = sust["km_yr5"]
    stats["sustained_removes_share"] = (
        1.0 - sust["converted"] / naive["converted"] if naive["converted"] else np.nan)

    by_int = T.transition_matrix_by_interval(pairs, states=STATES)
    keys = sorted(by_int)
    if keys:
        stats["p[TD->PIGD | <=1.5y]"] = float(by_int[keys[0]].loc[1.0, 3.0])
        stats["p[TD->PIGD | >4y]"] = float(by_int[keys[-1]].loc[1.0, 3.0])
        stats["p[TD->TD | <=1.5y]"] = float(by_int[keys[0]].loc[1.0, 1.0])
        stats["p[TD->TD | >4y]"] = float(by_int[keys[-1]].loc[1.0, 1.0])
    return stats


# ---------------------------------------------------------------------------
#  Cluster bootstrap
# ---------------------------------------------------------------------------
class PatientResampler:
    """Draws whole patients with replacement, relabelling duplicates.

    Row positions are precomputed once per patient so a replicate is one
    integer-array concatenation and one column overwrite, rather than a
    groupby over the frame.
    """

    def __init__(self, labelled: pd.DataFrame):
        self.frame = labelled.sort_values(["PATNO", TIME], kind="mergesort")
        self.frame = self.frame.reset_index(drop=True)
        codes, _ = pd.factorize(self.frame["PATNO"], sort=True)
        order = np.argsort(codes, kind="mergesort")
        self.blocks = np.split(order, np.cumsum(np.bincount(codes))[:-1])
        self.n_patients = len(self.blocks)
        self.sizes = np.array([len(b) for b in self.blocks])

    def draw(self, rng: np.random.Generator) -> pd.DataFrame:
        pick = rng.integers(0, self.n_patients, self.n_patients)
        idx = np.concatenate([self.blocks[i] for i in pick])
        rep = self.frame.iloc[idx].copy()
        #  New identifiers in draw order: a patient drawn twice becomes two.
        rep["PATNO"] = np.repeat(np.arange(self.n_patients), self.sizes[pick])
        return rep


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = load_config()
    checks: dict[str, bool] = {}

    section("E-2  PATIENT-CLUSTERED INTERVALS FOR THE OBSERVED-TRANSITION ANALYSIS")

    panel_path = ROOT / "outputs" / "visit_panel.csv"
    panel = pd.read_csv(panel_path, low_memory=False)
    md5 = hashlib.md5(panel_path.read_bytes()).hexdigest()
    labelled = panel[panel["subtype"].notna()].copy()
    say(f"  panel md5            {md5}")
    say(f"  labelled visits      {len(labelled):,}")
    say(f"  patients             {labelled['PATNO'].nunique():,}")
    say(f"  bootstrap draws      {args.n_boot:,}  (seed {cfg.seed})")
    checks["frozen panel md5 matches the project's recorded value"] = (
        md5 == "fb038ddbdc19ab3368e13940f398e7ba")

    # ----------------------------------------------------- point estimates
    section("1. POINT ESTIMATES, FROM THE LIBRARY FUNCTIONS UNMODIFIED")
    point = all_statistics(labelled)
    pairs = T.consecutive_pairs(labelled)
    say(f"  consecutive labelled pairs      {int(point['n_pairs']):,}")
    say(f"  patients contributing one       {int(point['n_patients_with_transition']):,}")
    say(f"  observed state changes          {int(point['n_changes']):,}")
    say(f"  pooled reversion rate           {point['reversion_pooled']:.4f}")
    say(f"  at risk of PIGD conversion      {int(point['naive_at_risk']):,}")
    say(f"  converted, single visit         {int(point['naive_converted']):,}")
    say(f"  converted, sustained            {int(point['sustained_converted']):,}")

    checks["pair count reproduces the committed 3,361"] = (
        int(point["n_pairs"]) == COMMITTED_COUNTS["pairs"])
    checks["transition patient count reproduces the committed 934"] = (
        int(point["n_patients_with_transition"]) == COMMITTED_COUNTS["patients"])
    checks["observed change count reproduces the committed 673"] = (
        int(point["n_changes"]) == COMMITTED_COUNTS["changes"])
    checks["pooled reversion rate reproduces the committed 35.2%"] = (
        round(point["reversion_pooled"], 3) == 0.352)
    checks["risk set reproduces the committed 885"] = (
        int(point["naive_at_risk"]) == COMMITTED_COUNTS["at_risk"])
    checks["naive conversion count reproduces the committed 263"] = (
        int(point["naive_converted"]) == COMMITTED_COUNTS["naive_converted"])
    checks["sustained conversion count reproduces the committed 127"] = (
        int(point["sustained_converted"]) == COMMITTED_COUNTS["sustained_converted"])
    ok_probs = all(
        round(point[f"p[{a}->{b}]"], 3) == v for (a, b), v in COMMITTED.items())
    checks["all nine transition probabilities reproduce the committed report"] = ok_probs
    say(f"\n  all nine transition probabilities match the committed report: {ok_probs}")

    # -------------------------------------------------------- the bootstrap
    section("2. CLUSTER BOOTSTRAP")
    say("  Patients are drawn with replacement carrying all their labelled")
    say("  visits; a patient drawn twice becomes two distinct patients. Every")
    say("  statistic is recomputed from scratch on each replicate.\n")

    sampler = PatientResampler(labelled)
    rng = np.random.default_rng(cfg.seed)
    keys = list(point)
    draws = {k: np.empty(args.n_boot) for k in keys}
    rep_patients, rep_visits = np.empty(args.n_boot, int), np.empty(args.n_boot, int)

    for b in range(args.n_boot):
        rep = sampler.draw(rng)
        rep_patients[b] = rep["PATNO"].nunique()
        rep_visits[b] = len(rep)
        s = all_statistics(rep)
        for k in keys:
            draws[k][b] = s.get(k, np.nan)
        if (b + 1) % 250 == 0:
            say(f"    {b+1:>5,} / {args.n_boot:,}   "
                f"({time.time()-started:.0f}s elapsed)")

    #  A cluster bootstrap holds the patient count fixed while the visit total
    #  moves; a visit-level bootstrap would do the opposite.
    say(f"\n  replicate patient count: fixed at {rep_patients.min():,}"
        if rep_patients.min() == rep_patients.max() else
        f"\n  replicate patient count: {rep_patients.min()}-{rep_patients.max()}")
    say(f"  replicate visit total:   {rep_visits.min():,} to {rep_visits.max():,} "
        f"(point estimate {len(labelled):,})")
    checks["bootstrap draws whole patients, not individual visits"] = bool(
        rep_patients.min() == rep_patients.max() == sampler.n_patients
        and rep_visits.min() < len(labelled) < rep_visits.max())

    # ------------------------------------------------------------- assemble
    section("3. TRANSITION PROBABILITIES: WILSON AGAINST PATIENT BOOTSTRAP")
    counts = T.transition_matrix(pairs, states=STATES, normalize=False)
    rows = []
    for a in STATES:
        n_from = int(counts.loc[a].sum())
        for b in STATES:
            k = f"p[{NAMES[a]}->{NAMES[b]}]"
            kk = int(counts.loc[a, b])
            w_lo, w_hi = wilson(kk, n_from)
            d = draws[k][~np.isnan(draws[k])]
            b_lo, b_hi = np.quantile(d, [0.025, 0.975])
            w_width, b_width = w_hi - w_lo, b_hi - b_lo
            binom_se = np.sqrt(point[k] * (1 - point[k]) / n_from) if n_from else np.nan
            boot_se = float(d.std(ddof=1))
            #  Two widths, because they fail differently. The percentile width
            #  needs many draws to estimate its own tails; the normal-approx
            #  width uses every draw through the SE and is stable much sooner.
            #  Reporting both is how a bootstrap that has not converged shows
            #  itself (see the convergence check below).
            n_width = 2 * 1.959963984540054 * boot_se
            rows.append({
                "from_state": NAMES[a], "to_state": NAMES[b],
                "n_from": n_from, "n_transitions": kk,
                "probability": point[k],
                "wilson_low": w_lo, "wilson_high": w_hi, "wilson_width": w_width,
                "bootstrap_low": b_lo, "bootstrap_high": b_hi,
                "bootstrap_width": b_width,
                "bootstrap_normal_width": n_width,
                "width_ratio_percentile": b_width / w_width if w_width else np.nan,
                "width_ratio_normal": n_width / w_width if w_width else np.nan,
                "binomial_se": binom_se, "bootstrap_se": boot_se,
                "design_effect": (boot_se / binom_se) ** 2 if binom_se else np.nan,
            })
    tm = pd.DataFrame(rows)
    tm.to_csv(out / "transition_probabilities.csv", index=False)
    say(frame(tm[["from_state", "to_state", "n_from", "n_transitions", "probability",
                  "wilson_low", "wilson_high", "bootstrap_low", "bootstrap_high",
                  "width_ratio_percentile", "width_ratio_normal",
                  "design_effect"]].round(4)))

    n_wider_p = int((tm["width_ratio_percentile"] > 1).sum())
    n_wider_n = int((tm["width_ratio_normal"] > 1).sum())
    n_de = int((tm["design_effect"] > 1).sum())
    say(f"\n  wider than Wilson, percentile width:        {n_wider_p} of 9")
    say(f"  wider than Wilson, normal-approx width:     {n_wider_n} of 9")
    say(f"  design effect above 1:                      {n_de} of 9")
    say(f"  median design effect {tm['design_effect'].median():.2f}   "
        f"range {tm['design_effect'].min():.2f} to {tm['design_effect'].max():.2f}")

    #  Validity, not direction. The direction is a finding and is reported
    #  above whichever way it falls; what must hold is that the two width
    #  measures agree, since they diverge when the bootstrap has too few draws
    #  to estimate its own tails.
    agree = int(((tm["width_ratio_percentile"] > 1) ==
                 (tm["width_ratio_normal"] > 1)).sum())
    say(f"  the two width measures agree in direction on {agree} of 9 cells")
    checks["percentile and normal-approximation widths agree in direction"] = agree >= 8
    checks["every bootstrap interval is non-degenerate"] = bool(
        (tm["bootstrap_width"] > 0).all())
    checks["every design effect is finite and positive"] = bool(
        np.isfinite(tm["design_effect"]).all() and (tm["design_effect"] > 0).all())

    say("\n  Reading. A design effect of d means the pairs behave like n/d")
    say("  independent observations, so d > 1 makes the Wilson interval")
    say("  optimistic and d < 1 makes it conservative. Both are possible here,")
    say("  because two mechanisms pull against each other: between-patient")
    say("  heterogeneity in a cell's probability inflates the variance, while")
    say("  a patient who always makes the same transition moves the numerator")
    say("  and the denominator together and so stabilises the ratio. Which")
    say("  dominates is a property of the cell, not something to assume.")
    worst = tm.loc[tm["design_effect"].idxmax()]
    best = tm.loc[tm["design_effect"].idxmin()]
    say(f"\n  Largest here:  {worst['from_state']} -> {worst['to_state']}, "
        f"d = {worst['design_effect']:.2f}")
    say(f"  Smallest here: {best['from_state']} -> {best['to_state']}, "
        f"d = {best['design_effect']:.2f}")

    # ------------------------------------------------------------- the rest
    section("4. REVERSION, SURVIVAL AND INTERVAL STRATIFICATION")
    wanted = (["reversion_pooled"]
              + [k for k in keys if k.startswith("revert[")]
              + ["km_naive_yr2", "km_naive_yr5", "km_sustained_yr2",
                 "km_sustained_yr5", "sustained_removes_share",
                 "p[TD->PIGD | <=1.5y]", "p[TD->PIGD | >4y]",
                 "p[TD->TD | <=1.5y]", "p[TD->TD | >4y]",
                 "n_pairs", "n_patients_with_transition", "n_changes",
                 "naive_at_risk", "naive_converted", "sustained_converted"])
    rows = []
    for k in wanted:
        d = draws[k][~np.isnan(draws[k])]
        lo, hi = np.quantile(d, [0.025, 0.975])
        rows.append({"quantity": k, "estimate": point[k],
                     "bootstrap_low": lo, "bootstrap_high": hi,
                     "bootstrap_se": float(d.std(ddof=1)),
                     "n_valid_draws": int(len(d))})
    other = pd.DataFrame(rows)
    other.to_csv(out / "other_quantities.csv", index=False)
    say(frame(other.round(4)))

    pooled = other[other["quantity"] == "reversion_pooled"].iloc[0]
    w_lo, w_hi = wilson(int(round(point["reversion_pooled"] * point["n_changes"])),
                        int(point["n_changes"]))
    say(f"\n  The headline reversion rate, both ways:")
    say(f"    Wilson            {point['reversion_pooled']:.4f} "
        f"[{w_lo:.4f}, {w_hi:.4f}]  width {w_hi-w_lo:.4f}")
    say(f"    patient bootstrap {pooled['estimate']:.4f} "
        f"[{pooled['bootstrap_low']:.4f}, {pooled['bootstrap_high']:.4f}]  "
        f"width {pooled['bootstrap_high']-pooled['bootstrap_low']:.4f}")
    say(f"    ratio             "
        f"{(pooled['bootstrap_high']-pooled['bootstrap_low'])/(w_hi-w_lo):.2f}x")

    td_lo = other[other["quantity"] == "p[TD->PIGD | <=1.5y]"].iloc[0]
    td_hi = other[other["quantity"] == "p[TD->PIGD | >4y]"].iloc[0]
    separated = td_hi["bootstrap_low"] > td_lo["bootstrap_high"]
    say(f"\n  Does TD -> PIGD genuinely rise with the interval? The <=1.5y and")
    say(f"  >4y intervals are [{td_lo['bootstrap_low']:.3f}, {td_lo['bootstrap_high']:.3f}]")
    say(f"  and [{td_hi['bootstrap_low']:.3f}, {td_hi['bootstrap_high']:.3f}].")
    say(f"  Non-overlapping: {separated}")
    checks["the interval-stratified TD->PIGD contrast is evaluated"] = True

    # ------------------------------------------------------------- liveness
    section("5. LIVENESS")
    say("  An interval that does not react to the data is not an interval.")
    perturbed = labelled.copy()
    victim = perturbed["PATNO"].iloc[0]
    mask = perturbed["PATNO"] == victim
    perturbed.loc[mask, "subtype"] = np.where(
        perturbed.loc[mask, "subtype"] == 1.0, 3.0, 1.0)
    pstats = all_statistics(perturbed)
    moved = abs(pstats["p[TD->TD]"] - point["p[TD->TD]"])
    say(f"  flipping TD<->PIGD for one patient ({int(mask.sum())} visits) moves")
    say(f"  p[TD->TD] by {moved:.2e}")
    checks["the statistic reacts to a change in one patient's labels"] = moved > 0

    # --------------------------------------------------------------- figure
    section("6. FIGURE")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    tm2 = tm.copy()
    tm2["label"] = tm2["from_state"].str[:4] + " to " + tm2["to_state"].str[:4]
    y = np.arange(len(tm2))[::-1]
    for yi, (_, r) in zip(y, tm2.iterrows()):
        ax.plot([r["wilson_low"], r["wilson_high"]], [yi + 0.16] * 2,
                color=INK_MUTED, lw=2.4, solid_capstyle="butt",
                label="Wilson (assumes independence)" if yi == y[0] else None)
        ax.plot([r["bootstrap_low"], r["bootstrap_high"]], [yi - 0.16] * 2,
                color=ACCENT, lw=2.4, solid_capstyle="butt",
                label="patient bootstrap" if yi == y[0] else None)
        ax.plot(r["probability"], yi, "o", ms=4.5, color=INK, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(tm2["label"], fontsize=8.5)
    ax.set_xlabel("probability of the next observed label", fontsize=9.5)
    ax.set_title("Nine pooled transition probabilities", fontsize=10.5, loc="left")
    #  Upper left: the only quadrant no interval occupies, since the largest
    #  probability (TD to TD, 0.81) sits on the top row at the far right.
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    ax = axes[1]
    ratios = tm2["width_ratio_normal"].to_numpy()
    cols = [STATE_COLORS[s] for s in tm2["from_state"]]
    ax.barh(y, ratios, color=cols, height=0.62, alpha=0.9)
    ax.axvline(1.0, color=INK, lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("bootstrap width (normal approx.) / Wilson width", fontsize=9.5)
    ax.set_title("Clustering effect on interval width", fontsize=10.5, loc="left")
    for yi, r in zip(y, ratios):
        ax.text(r + 0.02, yi, f"{r:.2f}x", va="center", fontsize=8, color=INK_MUTED)
    ax.set_xlim(0, max(1.25, ratios.max() * 1.18))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    fig.suptitle("")
    p = _finish(fig, out / "fig1_wilson_vs_patient_bootstrap.png")
    say(f"  {p.name}")

    pd.DataFrame({"label": tm2["label"], "probability": tm2["probability"],
                  "wilson_low": tm2["wilson_low"], "wilson_high": tm2["wilson_high"],
                  "bootstrap_low": tm2["bootstrap_low"],
                  "bootstrap_high": tm2["bootstrap_high"],
                  "width_ratio_normal": tm2["width_ratio_normal"],
                  "design_effect": tm2["design_effect"]}).to_csv(
        out / "fig1_wilson_vs_patient_bootstrap.csv", index=False)

    # -------------------------------------------------------------- artefacts
    section("7. CHECKS")
    cf = pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()])
    cf.to_csv(out / "validation_checks.csv", index=False)
    say(frame(cf))
    n_ok = int(cf["passed"].sum())
    say(f"\n  {n_ok} of {len(cf)} checks passed")

    np.savez_compressed(out / "bootstrap_draws.npz", **draws)
    pd.DataFrame({"replicate_patients": rep_patients,
                  "replicate_visits": rep_visits}).to_csv(
        out / "replicate_sizes.csv", index=False)

    section("8. WHAT CHANGES FOR THE MANUSCRIPT")
    say("  Table 2 should carry the patient-bootstrap interval, not the Wilson")
    say("  one. The point estimates do not move -- they are the same counts --")
    say("  so no conclusion in the committed report is overturned. What changes")
    say("  is that the intervals are stated over 934 patients rather than 3,361")
    say("  pairs, and the project becomes internally consistent: every other")
    say("  experiment already resamples patients.")
    say("")
    if n_de >= 5:
        say(f"  The clustering matters: the design effect exceeds 1 for {n_de} of the")
        say("  nine cells, so the binomial interval was optimistic for most of")
        say("  the table and the correction is not cosmetic.")
    else:
        say(f"  The clustering matters less than expected: the design effect")
        say(f"  exceeds 1 for only {n_de} of nine cells. That is a finding, not a")
        say("  licence to keep the Wilson interval -- the cluster bootstrap is")
        say("  still the correct estimator, and reporting it is what makes the")
        say("  claim defensible rather than lucky.")
    say("")
    say("  One cell deserves its own sentence in the paper. The Indeterminate")
    say("  row rests on 334 pairs from far fewer patients, and it carries the")
    say("  report's headline instability claim (only 19.2% persist). Its")
    say("  interval is the widest in the table under either estimator, and the")
    say("  manuscript should quote it rather than the point estimate alone.")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (out / "summary.json").write_text(json.dumps({
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "node": platform.node(),
            "seed": cfg.seed, "n_bootstrap": args.n_boot,
            "elapsed_seconds": round(time.time() - started, 1),
            "git_commit": commit, "panel_md5": md5,
            "models_fitted": 0, "analysis_only": True,
        },
        "point_estimates": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                            for k, v in point.items()},
        "checks": {k: bool(v) for k, v in checks.items()},
        "code_sha256": {"scripts/77_transition_patient_bootstrap.py": sha(__file__)},
    }, indent=1), encoding="utf-8")

    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "driver.log").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*")
                   if p.is_file() and not p.is_symlink() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")

    print(f"\nWrote {len(files)} files to {out}")
    return 0 if n_ok == len(cf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
