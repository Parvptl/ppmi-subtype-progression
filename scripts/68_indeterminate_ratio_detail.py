"""Detailed scrutiny of the two commonest Indeterminate ratios, 0.9091 and 1.1364.

Script 67 asked whether the forced TD/PIGD call tracks the nearer boundary over
all Indeterminate visits and found no evidence that it does. This script looks
specifically at the two ratio values that dominate the band:

    0.9091  (n=144)  0.0091 above the PIGD boundary -> nearer PIGD
    1.1364  (n=67)   0.0136 below the TD boundary   -> nearer TD

Together they are 211 of the 287 Indeterminate visits with a defined ratio.

Everything here reuses the **existing held-out** predictions. No model is
refitted, and nothing in-sample is substituted: the CT-HMM's predictions come
from leave-one-patient-out, where every row's `held_out_patient` equals its
`PATNO`, and the rich profile's come from the first cross-validation repeat,
where each patient appears in exactly one fold. Both are asserted below.

The central test is **not** "does each group agree with its own nearer
boundary", because that statistic is confounded by the fact that both models
predict TD most of the time. It is the **difference between the two groups**:

    P(predict PIGD | ratio 0.9091)  -  P(predict PIGD | ratio 1.1364)

which should be positive if the calls respond to boundary proximity, and which
a constant-prediction model would score at exactly zero. The same contrast is
run on the predicted probabilities, which do not depend on a threshold.

All uncertainty is patient-clustered: patients, not visits, are resampled,
because a patient can contribute several visits and may contribute to both
ratio groups.

    python scripts/68_indeterminate_ratio_detail.py --out <dir>
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

from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import (  # noqa: E402
    GRID, INK, INK_MUTED, STATE_COLORS, SURFACE, _finish, _style,
)

CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
RP_RUN = "experiments/20261020T180515Z_clinical_profile_cv"
PRIOR_RUN = "experiments/20261023T203431Z_indeterminate_boundary"
CT_MODEL = "CT-HMM (latent, interval-aware)"
RP_MODEL = "Profile: level + rate + medication"

#: The two focal values and the two comparison values requested.
FOCAL = (0.9091, 1.1364)
COMPARISON = (1.0227, 1.0606)
ALL_FOUR = (0.9091, 1.0227, 1.0606, 1.1364)

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


# ---------------------------------------------------------------------------
#  Patient-clustered bootstrap
# ---------------------------------------------------------------------------
def patient_boot(frame: pd.DataFrame, stat, seed: int = 42, n_boot: int = 4000):
    """Percentile CI for `stat(frame)`, resampling PATIENTS with replacement.

    The statistic is handed the resampled frame, so a statistic that spans
    both ratio groups (a difference between them) is resampled coherently: a
    patient contributing visits at both ratios is drawn, or not drawn, as a
    whole. Draws where the statistic is undefined -- a resample containing no
    visits from one of the groups -- are discarded and counted.
    """
    groups = {p: g for p, g in frame.groupby("PATNO", sort=False)}
    patients = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        sample = pd.concat([groups[patients[i]] for i in pick], ignore_index=True)
        try:
            draws[b] = stat(sample)
        except (ZeroDivisionError, ValueError, KeyError):
            continue
    ok = draws[np.isfinite(draws)]
    if not len(ok):
        return float("nan"), float("nan"), 0
    return (float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975)),
            int(len(ok)))


def share_pigd(d: pd.DataFrame, col: str) -> float:
    if not len(d):
        raise ValueError("empty")
    return float((d[col] == "PIGD").mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()
    LO, HI = cfg.subtype.pigd_threshold, cfg.subtype.td_threshold

    # ================================================= 1. assemble and verify
    section("1. VISIT-LEVEL RECORDS, AND VERIFICATION AGAINST SCRIPT 67")

    panel = pd.read_csv(ROOT / "outputs/visit_panel.csv", low_memory=False)
    meas = panel[["PATNO", "EVENT_ID", "visit_date", "years_from_baseline",
                  "tremor_score", "pigd_score", "ratio", "ratio_defined"]].copy()

    ct = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv", low_memory=False)
    ct = ct[ct["model"] == CT_MODEL].copy()
    rp = pd.read_csv(ROOT / RP_RUN / "derived_label_predictions.csv", low_memory=False)
    rp = rp[rp["model"] == RP_MODEL].copy()
    prof = pd.read_csv(ROOT / RP_RUN / "profile_predictions.csv", low_memory=False)
    prof = prof[(prof["model"] == RP_MODEL) & (prof["repeat"] == 0)]

    # Held-out provenance, asserted rather than assumed.
    assert (ct["held_out_patient"] == ct["PATNO"]).all(), \
        "CT-HMM rows are not all leave-one-patient-out"
    say("  CT-HMM   : leave-one-patient-out; held_out_patient == PATNO on every "
        "row [OK]")
    folds = prof.groupby("PATNO")["fold"].nunique()
    assert (folds == 1).all(), "a patient appears in more than one fold"
    say("  Profile  : cross-validation repeat 0; each patient in exactly one "
        "fold [OK]")

    key = ["PATNO", "current_event", "target_event"]
    shared = (set(map(tuple, ct[key].to_numpy()))
              & set(map(tuple, rp[key].to_numpy())))

    def prep(d):
        d = d[[tuple(r) in shared for r in d[key].to_numpy()]].copy()
        d = d[d["actual_next_label"] == "Indeterminate"]
        return d.sort_values(key).reset_index(drop=True)

    ct_i, rp_i = prep(ct), prep(rp)
    assert (ct_i[key].to_numpy() == rp_i[key].to_numpy()).all(), "row alignment"

    rec = ct_i[["PATNO", "current_event", "target_event", "current_label",
                "actual_next_label", "interval_years", "visit_index"]].copy()
    rec["ct_predicted"] = ct_i["predicted_state"].to_numpy()
    rec["ct_P_TD"] = ct_i["P_TD"].to_numpy()
    rec["ct_P_PIGD"] = ct_i["P_PIGD"].to_numpy()
    rec["rp_predicted"] = rp_i["predicted_state"].to_numpy()
    rec["rp_P_TD"] = rp_i["P_TD"].to_numpy()
    rec["rp_P_PIGD"] = rp_i["P_PIGD"].to_numpy()
    rec["ct_heldout"] = True
    rec = rec.merge(prof[["PATNO", "current_event", "target_event", "fold"]],
                    on=key, how="left").rename(columns={"fold": "rp_cv_fold"})
    rec = rec.merge(meas, left_on=["PATNO", "target_event"],
                    right_on=["PATNO", "EVENT_ID"], how="left")

    say(f"\n  Indeterminate-target visits shared by both models : {len(rec):,}")
    undef = int((~rec["ratio_defined"].astype(bool)).sum())
    say(f"    ratio undefined (PIGD subscore zero), excluded  : {undef}")
    rec = rec[rec["ratio_defined"].astype(bool)].reset_index(drop=True)
    rec["ratio_r"] = rec["ratio"].round(4)
    rec["dist_to_PIGD"] = rec["ratio"] - LO
    rec["dist_to_TD"] = HI - rec["ratio"]
    rec["nearer_boundary"] = np.where(rec["dist_to_PIGD"] < rec["dist_to_TD"],
                                      "PIGD", "TD")
    say(f"    ANALYSED                                        : {len(rec):,}"
        f"  ({rec['PATNO'].nunique():,} patients)")

    # Reproduce script 67's distribution exactly.
    prior = pd.read_csv(ROOT / PRIOR_RUN / "ratio_values.csv")
    mine = (rec["ratio_r"].value_counts().sort_index()
            .rename_axis("ratio").reset_index(name="n"))
    merged = prior[["ratio", "n"]].merge(mine, on="ratio", how="outer",
                                         suffixes=("_script67", "_here"))
    ok = bool((merged["n_script67"] == merged["n_here"]).all())
    say(f"\n  ratio distribution reproduces script 67 exactly : {ok}")
    if not ok:
        say(block(merged, index=False))
        raise SystemExit("ratio counts disagree with the previous analysis")

    missing = {c: int(rec[c].isna().sum()) for c in
               ("visit_date", "ratio", "ct_P_PIGD", "rp_P_PIGD", "rp_cv_fold")}
    say(f"  missing values among recovered fields           : {missing}")
    rec.to_csv(out / "visit_level_records.csv", index=False)
    say(f"  wrote visit_level_records.csv ({len(rec):,} rows, "
        f"{len(rec.columns)} columns)")

    # ================================================ 2. the two focal ratios
    section("2. THE TWO FOCAL RATIOS, EACH MODEL SEPARATELY")
    say(f"  0.9091 sits {0.9091-LO:.4f} above the PIGD boundary -> nearer PIGD")
    say(f"  1.1364 sits {HI-1.1364:.4f} below the TD boundary   -> nearer TD")

    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = []
    for r in ALL_FOUR:
        g = rec[rec["ratio_r"] == r]
        near = "PIGD" if r < (LO + HI) / 2 else "TD"
        for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile")):
            pred = g[f"{tag}_predicted"]
            pp = g[f"{tag}_P_PIGD"].to_numpy(float)
            pt = g[f"{tag}_P_TD"].to_numpy(float)
            n_td, n_pg = int((pred == "TD").sum()), int((pred == "PIGD").sum())
            row = {"ratio": r, "nearer_boundary": near, "model": name,
                   "n_visits": len(g), "n_patients": int(g["PATNO"].nunique()),
                   "n_predicted_TD": n_td, "n_predicted_PIGD": n_pg,
                   "pct_predicted_TD": n_td / len(g) if len(g) else np.nan,
                   "pct_predicted_PIGD": n_pg / len(g) if len(g) else np.nan,
                   "n_matching_nearer": int((pred == near).sum()),
                   "pct_matching_nearer": float((pred == near).mean()) if len(g) else np.nan,
                   "mean_P_PIGD": float(pp.mean()), "median_P_PIGD": float(np.median(pp)),
                   "mean_P_TD": float(pt.mean()), "median_P_TD": float(np.median(pt))}
            for q in qs:
                row[f"P_PIGD_q{int(q*100)}"] = float(np.quantile(pp, q))
            rows.append(row)
    per_ratio = pd.DataFrame(rows)
    per_ratio.round(4).to_csv(out / "per_ratio_tables.csv", index=False)

    for r in FOCAL:
        near = "PIGD" if r < (LO + HI) / 2 else "TD"
        sub = per_ratio[per_ratio["ratio"] == r]
        say(f"\n  --- ratio {r}  (nearer {near} boundary) ---")
        say(block(sub[["model", "n_visits", "n_patients", "n_predicted_TD",
                       "n_predicted_PIGD", "pct_predicted_PIGD",
                       "n_matching_nearer", "pct_matching_nearer",
                       "mean_P_PIGD", "median_P_PIGD", "mean_P_TD",
                       "median_P_TD"]].round(4), index=False))
        say("    predicted-PIGD-probability quantiles:")
        say(block(sub[["model"] + [f"P_PIGD_q{int(q*100)}" for q in qs]].round(4),
                  index=False))

    # ============================================ 3. all four ratios together
    section("3. ALL FOUR RATIO VALUES COMPARED")
    say(block(per_ratio[["ratio", "nearer_boundary", "model", "n_visits",
                         "n_patients", "pct_predicted_PIGD", "mean_P_PIGD",
                         "median_P_PIGD"]].round(4), index=False))
    say("")
    say("  The four observed values are discrete and very unevenly represented")
    say("  (144, 16, 35, 67). This is NOT a continuous, uniformly sampled ratio")
    say("  distribution, and no trend line should be read across them.")

    # ================================================= 4. statistical testing
    section("4. PATIENT-CLUSTERED UNCERTAINTY")
    say("  Resampling unit: PATIENT. Patients are drawn with replacement and")
    say("  all of their visits travel together, because a patient can")
    say("  contribute several visits and may appear at both focal ratios.")
    focal = rec[rec["ratio_r"].isin(FOCAL)].copy()
    say(f"\n  focal set: {len(focal):,} visits, "
        f"{focal['PATNO'].nunique():,} unique patients")
    overlap = (focal.groupby("PATNO")["ratio_r"].nunique() > 1).sum()
    say(f"  patients contributing visits at BOTH focal ratios: {int(overlap)}")

    stats = []

    # (a) the two headline proportions
    for r, target, label in ((0.9091, "PIGD", "P(predict PIGD | ratio 0.9091)"),
                             (1.1364, "TD", "P(predict TD | ratio 1.1364)")):
        g = rec[rec["ratio_r"] == r]
        for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile")):
            est = float((g[f"{tag}_predicted"] == target).mean())
            lo, hi, nd = patient_boot(
                g, lambda d, t=tag, tg=target: float((d[f"{t}_predicted"] == tg).mean()),
                n_boot=args.n_boot)
            stats.append({"quantity": label, "model": name, "estimate": est,
                          "ci_low": lo, "ci_high": hi, "n_visits": len(g),
                          "n_patients": int(g["PATNO"].nunique()), "n_draws": nd})

    # (b) the boundary-sensitivity contrast, per model
    def contrast_label(d, tag):
        a = d[d["ratio_r"] == 0.9091]
        b = d[d["ratio_r"] == 1.1364]
        if not len(a) or not len(b):
            raise ValueError("group missing")
        return share_pigd(a, f"{tag}_predicted") - share_pigd(b, f"{tag}_predicted")

    def contrast_prob(d, tag):
        a = d[d["ratio_r"] == 0.9091]
        b = d[d["ratio_r"] == 1.1364]
        if not len(a) or not len(b):
            raise ValueError("group missing")
        return float(a[f"{tag}_P_PIGD"].mean() - b[f"{tag}_P_PIGD"].mean())

    say("\n  Boundary sensitivity: the difference BETWEEN the two groups.")
    say("  A model whose calls ignore the ratio scores exactly zero here;")
    say("  a boundary-sensitive model scores positive.")
    for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile")):
        for fn, lab in ((contrast_label,
                         "P(PIGD | 0.9091) - P(PIGD | 1.1364)   [labels]"),
                        (contrast_prob,
                         "mean P_PIGD | 0.9091 - mean P_PIGD | 1.1364  [probs]")):
            est = fn(focal, tag)
            lo, hi, nd = patient_boot(focal, lambda d, f=fn, t=tag: f(d, t),
                                      n_boot=args.n_boot)
            stats.append({"quantity": lab, "model": name, "estimate": est,
                          "ci_low": lo, "ci_high": hi, "n_visits": len(focal),
                          "n_patients": int(focal["PATNO"].nunique()),
                          "n_draws": nd})

    # (c) paired model difference on the same visits
    for r in FOCAL:
        g = rec[rec["ratio_r"] == r]
        target = "PIGD" if r < (LO + HI) / 2 else "TD"
        est = float((g["rp_predicted"] == target).mean()
                    - (g["ct_predicted"] == target).mean())
        lo, hi, nd = patient_boot(
            g, lambda d, tg=target: float((d["rp_predicted"] == tg).mean()
                                          - (d["ct_predicted"] == tg).mean()),
            n_boot=args.n_boot)
        stats.append({"quantity": f"profile - CT-HMM, P(predict {target} | "
                                  f"ratio {r})  [paired]",
                      "model": "difference", "estimate": est, "ci_low": lo,
                      "ci_high": hi, "n_visits": len(g),
                      "n_patients": int(g["PATNO"].nunique()), "n_draws": nd})

    st = pd.DataFrame(stats)
    # Zero is the null only for the contrasts. For a raw proportion such as
    # P(predict PIGD | ratio 0.9091) the null is the model's own call rate, not
    # zero, so "excludes zero" there would be a meaningless flag and is left
    # blank rather than printed as though it were a test.
    st["is_contrast"] = st["quantity"].str.contains(" - ", regex=False)
    st["excludes_zero"] = np.where(
        st["is_contrast"], (st["ci_low"] > 0) | (st["ci_high"] < 0), None)
    st.round(4).to_csv(out / "patient_clustered_intervals.csv", index=False)
    say("")
    say(block(st[["quantity", "model", "estimate", "ci_low", "ci_high",
                  "n_visits", "n_patients", "excludes_zero"]].round(4),
              index=False, na_rep="   -"))
    say("")
    say("  'excludes_zero' is blank for the first four rows on purpose: they are")
    say("  raw proportions whose null is the model's own call rate, not zero.")
    say("  Only the contrasts below them are tests against a zero null.")
    n_tests = int(st["is_contrast"].sum())
    say(f"  {n_tests} of the {len(st)} intervals are such tests, and NO")
    say("  multiplicity correction has been applied. With this many comparisons,")
    say("  an isolated interval that just excludes zero is weak evidence alone.")

    # ======================================== 5. repeated-visit patients
    section("5. PATIENTS CONTRIBUTING SEVERAL FOCAL VISITS")
    counts = focal.groupby("PATNO").size().sort_values(ascending=False)
    say(f"  visits per patient in the focal set: "
        f"max {counts.max()}, median {int(counts.median())}, "
        f"{int((counts == 1).sum())} patients contribute exactly one")
    say(f"  share of focal visits from patients with >1 visit: "
        f"{float(counts[counts > 1].sum() / counts.sum()):.1%}")

    # Selection rule: among patients with MORE THAN ONE focal visit -- the only
    # ones who have a trajectory to look at -- take those with the most visits,
    # breaking ties by lowest patient id. The rule refers only to how many
    # visits a patient has, never to what either model predicted, so it cannot
    # favour examples that support the hypothesis.
    repeats = counts[counts > 1]
    chosen = list(sorted(repeats.index, key=lambda p: (-repeats[p], p))[:6])
    say(f"\n  example patients (rule: >1 focal visit, then most visits, ties by "
        f"lowest id;\n  the rule never looks at a prediction): {len(chosen)} of "
        f"{len(repeats)} eligible")

    anon = {p: f"P{idx + 1:02d}" for idx, p in enumerate(chosen)}
    # Show every Indeterminate visit these patients have, not only the focal
    # ones, so the trajectory is not truncated to the two values of interest.
    traj = rec[rec["PATNO"].isin(chosen)].sort_values(
        ["PATNO", "years_from_baseline"]).copy()
    traj["patient_anon"] = traj["PATNO"].map(anon)
    traj["is_focal"] = traj["ratio_r"].isin(FOCAL)
    cols = ["patient_anon", "PATNO", "target_event", "visit_date",
            "years_from_baseline", "ratio_r", "nearer_boundary", "is_focal",
            "ct_predicted", "ct_P_PIGD", "rp_predicted", "rp_P_PIGD"]
    traj[cols].round(4).to_csv(out / "example_trajectories.csv", index=False)
    say(block(traj[["patient_anon", "target_event", "years_from_baseline",
                    "ratio_r", "is_focal", "ct_predicted", "ct_P_PIGD",
                    "rp_predicted", "rp_P_PIGD"]].round(3), index=False))

    # Does a patient's ratio stay put across repeated focal visits?
    multi = counts[counts > 1].index
    by_pat = focal[focal["PATNO"].isin(multi)].groupby("PATNO")["ratio_r"]
    stable = int((by_pat.nunique() == 1).sum())
    say(f"\n  Of the {len(multi)} patients with >1 focal visit, {stable} stay at "
        f"a single\n  focal ratio across all of them and {len(multi) - stable} "
        f"move between 0.9091 and 1.1364.")

    # Within-patient variation when the ratio is unchanged.
    say("\n  Among patients with >1 focal visit at the SAME ratio:")
    same = focal.groupby(["PATNO", "ratio_r"]).filter(lambda g: len(g) > 1)
    if len(same):
        gg = same.groupby(["PATNO", "ratio_r"])
        spread = gg.agg(n=("ct_P_PIGD", "size"),
                        ct_range=("ct_P_PIGD", lambda s: s.max() - s.min()),
                        rp_range=("rp_P_PIGD", lambda s: s.max() - s.min()),
                        ct_flips=("ct_predicted", lambda s: s.nunique() > 1),
                        rp_flips=("rp_predicted", lambda s: s.nunique() > 1))
        say(f"    patient-ratio blocks with repeats : {len(spread)}")
        say(f"    CT-HMM  P_PIGD range, median      : {spread['ct_range'].median():.4f}"
            f"   max {spread['ct_range'].max():.4f}")
        say(f"    profile P_PIGD range, median      : {spread['rp_range'].median():.4f}"
            f"   max {spread['rp_range'].max():.4f}")
        say(f"    blocks where the CALL changes     : CT-HMM "
            f"{int(spread['ct_flips'].sum())}, profile "
            f"{int(spread['rp_flips'].sum())}  of {len(spread)}")
        spread.round(4).to_csv(out / "within_patient_same_ratio.csv")
    else:
        say("    none")

    agree = (focal["ct_predicted"] == focal["rp_predicted"])
    say(f"\n  models agree on the call for {int(agree.sum()):,} of "
        f"{len(focal):,} focal visits ({agree.mean():.1%})")

    # ================================================== 6. figures
    section("6. FIGURES")
    # (1) grouped bar chart at the two focal ratios
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _style(ax)
    x = np.arange(2)
    w = 0.2
    for k, (tag, name, colour) in enumerate((("ct", "CT-HMM", ACCENT),
                                             ("rp", "Rich profile", COOL))):
        for j, cls in enumerate(("TD", "PIGD")):
            vals, errs = [], [[], []]
            for r in FOCAL:
                g = rec[rec["ratio_r"] == r]
                v = float((g[f"{tag}_predicted"] == cls).mean())
                lo, hi, _ = patient_boot(
                    g, lambda d, t=tag, c=cls: float((d[f"{t}_predicted"] == c).mean()),
                    n_boot=1000)
                vals.append(v); errs[0].append(v - lo); errs[1].append(hi - v)
            pos = x + (k * 2 + j) * w - 1.5 * w
            ax.bar(pos, vals, w * 0.9, color=colour,
                   alpha=1.0 if cls == "PIGD" else 0.45,
                   label=f"{name} — predicted {cls}", zorder=3)
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK_MUTED,
                        elinewidth=1.1, capsize=3, zorder=4)
            for p_, v in zip(pos, vals):
                ax.text(p_, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.5,
                        color=INK)
    ax.set_xticks(x)
    n0 = int((rec["ratio_r"] == 0.9091).sum()); n1 = int((rec["ratio_r"] == 1.1364).sum())
    p0 = int(rec[rec['ratio_r'] == 0.9091]['PATNO'].nunique())
    p1 = int(rec[rec['ratio_r'] == 1.1364]['PATNO'].nunique())
    ax.set_xticklabels([f"ratio 0.9091 — nearer PIGD\nn = {n0} visits, {p0} patients",
                        f"ratio 1.1364 — nearer TD\nn = {n1} visits, {p1} patients"])
    ax.set_ylabel("Share of visits given each call")
    ax.set_ylim(0, 1.0)
    ax.set_title("Predicted subtype at the two commonest Indeterminate ratios\n"
                 "held-out predictions; bars are 95% patient-clustered intervals",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    p1f = _finish(fig, out / "fig1_calls_at_focal_ratios.png")
    say(f"  wrote {p1f.name}")

    # (2) predicted PIGD probability by observed ratio, per model
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    for ax, (tag, name) in zip(axes, (("ct", "CT-HMM"),
                                      ("rp", "Rich clinical profile"))):
        _style(ax)
        data, labels, ns = [], [], []
        for r in ALL_FOUR:
            g = rec[rec["ratio_r"] == r]
            data.append(g[f"{tag}_P_PIGD"].to_numpy(float))
            labels.append(f"{r}\n{'PIGD' if r < 1.025 else 'TD'}")
            ns.append((len(g), int(g["PATNO"].nunique())))
        pos = np.arange(len(data))
        bp = ax.boxplot(data, positions=pos, widths=0.5, showfliers=False,
                        patch_artist=True, medianprops=dict(color=INK, linewidth=1.6))
        for patch, r in zip(bp["boxes"], ALL_FOUR):
            patch.set_facecolor(STATE_COLORS["PIGD"] if r < 1.025
                                else STATE_COLORS["TD"])
            patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data):
            ax.scatter(pos[i] + rng.uniform(-0.14, 0.14, len(d)), d, s=9,
                       color=INK_MUTED, alpha=0.5, zorder=3)
        ax.axhline(0.5, color=WARM, linestyle="--", linewidth=1.3, zorder=2)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{l}\nn={n[0]}, {n[1]}p" for l, n in zip(labels, ns)],
                           fontsize=8.5)
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Predicted P(PIGD)")
    axes[0].text(0.02, 0.53, "decision threshold", transform=axes[0].transAxes,
                 fontsize=8, color=WARM, va="bottom")
    fig.suptitle("Predicted PIGD probability at each observed Indeterminate "
                 "ratio\nfour discrete values only — the x axis is categorical, "
                 "not continuous", fontsize=11, x=0.02, y=0.985, ha="left",
                 va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p2f = _finish(fig, out / "fig2_probability_by_ratio.png")
    say(f"  wrote {p2f.name}")

    # (3) patient trajectories, one small panel each so no lines cross
    if len(chosen):
        ncol = min(3, len(chosen))
        nrow = int(np.ceil(len(chosen) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.1 * nrow),
                                 sharey=True, squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for i, pat in enumerate(chosen):
            ax = axes[i // ncol][i % ncol]
            ax.set_visible(True)
            _style(ax)
            g = traj[traj["PATNO"] == pat].sort_values("years_from_baseline")
            xs = g["years_from_baseline"].to_numpy(float)
            for tag, colour, mk, nm in (("ct", ACCENT, "o", "CT-HMM"),
                                        ("rp", COOL, "s", "Rich profile")):
                ax.plot(xs, g[f"{tag}_P_PIGD"], marker=mk, color=colour,
                        linewidth=1.4, markersize=5.5, label=nm, zorder=3)
            ax.axhline(0.5, color=WARM, linestyle="--", linewidth=1.2, zorder=2)
            # mark the focal visits and label every visit with its ratio
            foc = g[g["is_focal"]]
            ax.scatter(foc["years_from_baseline"],
                       np.full(len(foc), 0.035), marker="^", s=26,
                       color=INK_MUTED, zorder=4)
            for x_, r_ in zip(xs, g["ratio_r"]):
                ax.annotate(f"{r_:.4f}", (x_, 0.97), ha="center", va="top",
                            fontsize=6.5, color=INK_MUTED, rotation=90)
            ax.set_title(f"{anon[pat]} — {len(g)} Indeterminate visits",
                         fontsize=9.5, loc="left")
            ax.set_ylim(0, 1)
            ax.margins(x=0.18)
            ax.grid(color=GRID, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            if i % ncol == 0:
                ax.set_ylabel("Predicted P(PIGD)")
            if i // ncol == nrow - 1:
                ax.set_xlabel("Years from baseline")
        axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("Predicted PIGD probability across a patient's "
                     "Indeterminate visits\nexamples chosen by visit count "
                     "alone; labels are the observed ratio, "
                     "▲ marks a focal visit (0.9091 or 1.1364)",
                     fontsize=11, x=0.01, y=0.995, ha="left", va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.94 if nrow > 1 else 0.86))
        p3f = _finish(fig, out / "fig3_patient_trajectories.png")
        say(f"  wrote {p3f.name}")

    # ================================================== 7. checks
    section("7. CHECKS")
    checks = {
        "ratio counts reproduce script 67": ok,
        "both models scored the identical visits":
            bool((ct_i[key].to_numpy() == rp_i[key].to_numpy()).all()),
        "CT-HMM rows are all leave-one-patient-out":
            bool((ct["held_out_patient"] == ct["PATNO"]).all()),
        "each patient in exactly one CV fold": bool((folds == 1).all()),
        "focal counts are 144 and 67":
            int((rec["ratio_r"] == 0.9091).sum()) == 144
            and int((rec["ratio_r"] == 1.1364).sum()) == 67,
        "focal visits total 211": len(focal) == 211,
        "every analysed visit has a defined ratio":
            bool(rec["ratio_defined"].astype(bool).all()),
        "all probabilities finite and in [0, 1]": bool(
            all(np.isfinite(rec[c]).all() and (rec[c] >= 0).all()
                and (rec[c] <= 1).all()
                for c in ("ct_P_TD", "ct_P_PIGD", "rp_P_TD", "rp_P_PIGD"))),
        "P_TD + P_PIGD == 1 for both models": bool(
            np.allclose(rec["ct_P_TD"] + rec["ct_P_PIGD"], 1.0, atol=1e-12)
            and np.allclose(rec["rp_P_TD"] + rec["rp_P_PIGD"], 1.0, atol=1e-12)),
        "no prediction is Indeterminate (binary by construction)":
            set(rec["ct_predicted"]) <= {"TD", "PIGD"}
            and set(rec["rp_predicted"]) <= {"TD", "PIGD"},
    }
    for n, v in checks.items():
        say(f"    [{'PASS' if v else 'FAIL'}]  {n}")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # ================================================== 8. provenance
    section("8. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "n_bootstrap": args.n_boot, "ct_hmm_run": CT_RUN,
           "profile_run": RP_RUN, "prior_boundary_run": PRIOR_RUN,
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
         "checks": checks, "missing_fields": missing,
         "n_excluded_undefined_ratio": undef,
         "per_ratio": per_ratio.to_dict(orient="records"),
         "intervals": st.to_dict(orient="records"),
         "example_patients": {str(k): v for k, v in anon.items()},
         "models_agree_share": float(agree.mean())},
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
