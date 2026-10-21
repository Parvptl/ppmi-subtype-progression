"""Part C1: raw tremor/PIGD item-sum audit, descriptives and sensitivity constructions.

Plan: docs/cluster_phase/10_PART_C_RESEARCH_PLAN.md (revision 2, approved).
Descriptive only; no model is fitted and no decision rule is applied.

    python scripts/39_raw_score_audit.py --out <new dir> [--smoke]

--smoke restricts every cohort to its first 150 patients by sorted PATNO and
uses 200 bootstrap replicates (code-path check only).

Reuses, unchanged: loaders, subtype (OFF selection, deduplication, cleaning,
classification), panel._add_time_axis, transitions.consecutive_pairs.
Cites: script 02 (transition matrix, reversion), script 08 (Indeterminate band
values), scripts 30/31 (PIGD measurement audit and models).
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import loaders  # noqa: E402
from ppmi_subtype import raw_scores as RS  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.transitions import consecutive_pairs  # noqa: E402

SEED = 42
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
PLAN = ROOT / "docs" / "cluster_phase" / "10_PART_C_RESEARCH_PLAN.md"
CODE = ["src/ppmi_subtype/raw_scores.py", "scripts/39_raw_score_audit.py",
        "src/ppmi_subtype/subtype.py", "src/ppmi_subtype/panel.py", "src/ppmi_subtype/loaders.py"]
LABELS = ["TD", "Indeterminate", "PIGD"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def baseline_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: first visit covariates, first labelled label, label counts."""
    f = frame.sort_values(["PATNO", "years_from_baseline"])
    first = f.groupby("PATNO").head(1).set_index("PATNO")
    lab = f[f.subtype_name.notna()]
    out = pd.DataFrame(index=first.index)
    for c in ("age", "SEX", "duration_yrs", "NHY", "PDTRTMNT"):
        if c in first:
            out[c] = pd.to_numeric(first[c], errors="coerce")
    out["baseline_label"] = lab.groupby("PATNO")["subtype_name"].first()
    out["n_labelled"] = lab.groupby("PATNO").size()
    out["n_labelled"] = out["n_labelled"].fillna(0).astype(int)
    out["n_visits_panel"] = f.groupby("PATNO").size()
    return out


def describe_groups(base: pd.DataFrame) -> pd.DataFrame:
    base = base.assign(group=np.where(base.n_labelled >= 2, "included (>=2 labelled)",
                                      np.where(base.n_labelled == 1, "1 labelled visit",
                                               "0 labelled visits")))
    rows = []
    for g, d in base.groupby("group"):
        r = {"group": g, "patients": len(d)}
        for c in ("age", "duration_yrs", "NHY"):
            if c in d:
                r[f"{c}_mean"] = d[c].mean()
                r[f"{c}_n"] = int(d[c].notna().sum())
        if "SEX" in d:
            r["SEX_1_share"] = (d.SEX == 1).mean()
        if "PDTRTMNT" in d:
            r["treated_at_first_visit_share"] = (d.PDTRTMNT == 1).mean()
        for L in LABELS:
            r[f"baseline_{L}_share"] = (d.baseline_label == L).mean()
        r["baseline_label_missing_share"] = d.baseline_label.isna().mean()
        rows.append(r)
    return pd.DataFrame(rows)


def visit_count_table(frame, cols):
    rows = []
    for col in cols:
        c = frame[frame[col].notna()].groupby("PATNO").size()
        rows.append({"score": col, "visits": int(c.sum()), "patients": len(c),
                     **{f">={k}": int((c >= k).sum()) for k in (2, 3, 5, 8)}})
    return pd.DataFrame(rows)


def yearly(frame, B, seed):
    """Yearly mean sums with patient-cluster bootstrap (year = floor(years from baseline))."""
    f = frame[frame.subtype_name.notna()].assign(yr=np.floor(frame.years_from_baseline))
    rows = []
    rng = np.random.default_rng(seed)
    for score in ("tremor_sum", "pigd_sum"):
        pats = np.sort(f.PATNO.unique())
        pidx = {p: i for i, p in enumerate(pats)}
        years = np.sort(f.yr.unique())
        yidx = {y: i for i, y in enumerate(years)}
        S = np.zeros((len(pats), len(years)))
        N = np.zeros((len(pats), len(years)))
        np.add.at(S, (f.PATNO.map(pidx), f.yr.map(yidx)), f[score].to_numpy())
        np.add.at(N, (f.PATNO.map(pidx), f.yr.map(yidx)), 1)
        bs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats))
            with np.errstate(invalid="ignore", divide="ignore"):
                bs.append((w @ S) / (w @ N))
        bs = np.array(bs)
        zero = f.assign(z=(f[score] == 0)).groupby("yr").z.mean()
        for y in years:
            i = yidx[y]
            rows.append({"score": score, "year": int(y), "n_visits": int(N[:, i].sum()),
                         "n_patients": int((N[:, i] > 0).sum()), "mean": S[:, i].sum() / N[:, i].sum(),
                         "ci_lo": np.nanquantile(bs[:, i], .025), "ci_hi": np.nanquantile(bs[:, i], .975),
                         "zero_share": zero.loc[y]})
    return pd.DataFrame(rows)


def event_kind(a, b):
    if a == b:
        return "stay"
    if b == "Indeterminate":
        return "entry"
    if a == "Indeterminate":
        return "exit"
    return "direct"


def consecutive_changes(frame):
    lab = frame[frame.subtype_name.notna()]
    pairs = consecutive_pairs(lab.assign(subtype=lab.subtype_name), label_col="subtype")
    keyed = lab.set_index(["PATNO", "years_from_baseline"])
    a = keyed.loc[list(zip(pairs.PATNO, pairs.from_time))][["tremor_sum", "pigd_sum"]].to_numpy()
    b = keyed.loc[list(zip(pairs.PATNO, pairs.to_time))][["tremor_sum", "pigd_sum"]].to_numpy()
    pairs["d_tremor"] = b[:, 0] - a[:, 0]
    pairs["d_pigd"] = b[:, 1] - a[:, 1]
    pairs["kind"] = [event_kind(x, y) for x, y in zip(pairs.from_state, pairs.to_state)]
    return pairs


def figures(out, frame, yearly_tab, examples, pairs, tq_edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fd / f"{name}.{ext}", dpi=150)
        plt.close(fig)

    lab = frame[frame.subtype_name.notna()].assign(yr=np.floor(frame.years_from_baseline))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, score, top in zip(axes, ("tremor_sum", "pigd_sum"), (20, 10)):
        yrs = [y for y in sorted(lab.yr.unique()) if (lab.yr == y).sum() >= 50]
        data = [np.clip(lab.loc[lab.yr == y, score], 0, top) for y in yrs]
        ax.boxplot(data, positions=yrs, widths=0.6, showfliers=False)
        z = lab.groupby("yr")[score].apply(lambda s: (s == 0).mean()).loc[yrs]
        ax2 = ax.twinx()
        ax2.plot(yrs, z, "r.-", label="share = 0")
        ax2.set_ylim(0, 0.5)
        ax2.set_ylabel("share of visits with sum = 0", color="r")
        ax.set_xlabel("years from first visit (floor); years with >= 50 visits")
        ax.set_ylabel(f"{score} (item points)")
    fig.suptitle("F1. Item-sum distributions by year (box: IQR, whiskers 1.5 IQR, no outliers)",
                 fontsize=10)
    save(fig, "F1_item_sum_distributions")

    pats = examples.sort_values(["baseline_label", "tertile", "PATNO"]).PATNO.tolist()
    if pats:
        nrow = int(np.ceil(len(pats) / 4))
        fig, axes = plt.subplots(nrow, 4, figsize=(15, 2.6 * nrow), squeeze=False)
        col = {"TD": "C0", "Indeterminate": "C2", "PIGD": "C3"}
        for ax, p in zip(axes.ravel(), pats):
            d = frame[frame.PATNO == p].sort_values("years_from_baseline")
            ax.plot(d.years_from_baseline, d.tremor_sum, "k-o", ms=3, lw=1, label="tremor sum")
            ax.plot(d.years_from_baseline, d.pigd_sum, "k--s", ms=3, lw=1, label="PIGD sum")
            for _, r in d.iterrows():
                x = r.years_from_baseline
                if pd.notna(r.subtype_name):
                    ax.axvspan(x - 0.15, x + 0.15, color=col[r.subtype_name], alpha=0.25, lw=0)
                    if r.pigd_sum == 0:
                        ax.text(x, -0.8, "P=0" if r.tremor_sum > 0 else "0/0", fontsize=6,
                                ha="center")
                else:
                    ax.text(x, -0.8, "x", color="0.4", fontsize=8, ha="center")
            ax.set_title(f"{d.subtype_name.dropna().iloc[0] if d.subtype_name.notna().any() else '?'}"
                         f" start, {int(d.subtype_name.notna().sum())} labelled visits", fontsize=8)
            ax.set_ylim(-1.5, max(6, np.nanmax(d[["tremor_sum", "pigd_sum"]].to_numpy()) + 1))
        for ax in axes.ravel()[len(pats):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("F2. Illustrative patients (seeded stratified draw; not for inference). "
                     "Shading = label (blue TD, green Indeterminate, red PIGD); "
                     "P=0: PIGD sum 0 (ratio undefined, TD by rule); 0/0: both zero "
                     "(Indeterminate by rule); x: visit without a label", fontsize=8)
        save(fig, "F2_example_patients")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, score in zip(axes, ("tremor_sum", "pigd_sum")):
        t = yearly_tab[(yearly_tab.score == score) & (yearly_tab.n_patients >= 30)]
        ax.errorbar(t.year, t["mean"], yerr=[t["mean"] - t.ci_lo, t.ci_hi - t["mean"]], fmt="o-")
        for _, r in t.iterrows():
            ax.annotate(str(r.n_patients), (r.year, r.ci_hi), fontsize=6, ha="center",
                        va="bottom")
        ax.set_xlabel("years from first visit (floor); labels = patients with a visit that year")
        ax.set_ylabel(f"mean {score} (item points)")
    fig.suptitle("F3. Yearly mean item sums, patient-bootstrap 95% intervals (years with >= 30 "
                 "patients). Attrition is informative (05_FINDINGS Stage 5).", fontsize=9)
    save(fig, "F3_yearly_means")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    rng = np.random.default_rng(SEED)
    for kind, c in (("stay", "0.75"), ("direct", "C3"), ("entry", "C2"), ("exit", "C0")):
        d = pairs[pairs.kind == kind]
        ax.scatter(d.d_tremor + rng.uniform(-.25, .25, len(d)),
                   d.d_pigd + rng.uniform(-.25, .25, len(d)), s=4, color=c, alpha=0.5,
                   label=f"{kind} (n={len(d)})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("change in tremor sum (item points), next labelled visit")
    ax.set_ylabel("change in PIGD sum (item points)")
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title("F4. Consecutive-visit score changes by label event (jittered)", fontsize=9)
    save(fig, "F4_consecutive_changes")

    grid = RS.label_grid(load_config().subtype)
    fig, ax = plt.subplots(figsize=(7, 5))
    H = np.zeros((26, 13))
    lab2 = lab[(lab.tremor_sum <= 25) & (lab.pigd_sum <= 12)]
    np.add.at(H, (lab2.tremor_sum.astype(int), lab2.pigd_sum.astype(int)), 1)
    ax.imshow(grid[:26, :13].T, origin="lower", cmap=matplotlib.colors.ListedColormap(
        ["#c6dbef", "#c7e9c0", "#fcbba1"]), vmin=1, vmax=3, aspect="auto")
    for (x, y), v in np.ndenumerate(H):
        if v:
            ax.text(x, y, int(v), ha="center", va="center", fontsize=5)
    ax.set_xlabel("tremor item sum (0-44; shown 0-25)")
    ax.set_ylabel("PIGD item sum (0-20; shown 0-12)")
    ax.set_title("F5. Label regions on the item-sum lattice (blue TD, green Indeterminate, "
                 "red PIGD)\nnumbers = labelled visits in each cell", fontsize=9)
    save(fig, "F5_label_lattice")


def run_cohort(name, frame, B, seed, out, tq_edges=None):
    """Tables common to a cohort; returns summary dict."""
    lab = frame[frame.subtype_name.notna()]
    base = baseline_table(frame)
    summary = {"patients": int(frame.PATNO.nunique()), "visits": int(len(frame)),
               "labelled_visits": int(len(lab)),
               "patients_ge2_labelled": int((base.n_labelled >= 2).sum()),
               "label_counts": lab.subtype_name.value_counts().to_dict(),
               "pigd_sum0_share": float((lab.pigd_sum == 0).mean()),
               "tremor_sum0_share": float((lab.tremor_sum == 0).mean())}
    describe_groups(base).to_csv(out / f"{name}_included_vs_excluded.csv", index=False)
    elig = base.groupby(base.baseline_label.fillna("none")).apply(
        lambda d: pd.Series({"patients": len(d), "ge2_labelled": int((d.n_labelled >= 2).sum())}),
        include_groups=False)
    elig.to_csv(out / f"{name}_eligibility_by_baseline_label.csv")
    frame.reason.value_counts().rename_axis("reason").reset_index(name="visits") \
        .to_csv(out / f"{name}_visit_reasons.csv", index=False)
    visit_count_table(frame, ["tremor_sum", "pigd_sum", "subtype_name"]) \
        .to_csv(out / f"{name}_visit_counts.csv", index=False)
    if tq_edges is None:
        tq_edges = RS.quintile_edges(lab.tremor_sum)
    f2 = lab.assign(pg=RS.pigd_group(lab.pigd_sum).values,
                    tq=RS.quintile_group(lab.tremor_sum, tq_edges).values)
    RS.transition_table(f2, "pg", RS.PIGD_GROUPS, B=B, seed=seed) \
        .to_csv(out / f"{name}_T3_pigd_transitions.csv", index=False)
    RS.transition_table(f2, "tq", [f"Q{i}" for i in range(1, 6)], B=B, seed=seed) \
        .to_csv(out / f"{name}_T3_tremor_transitions.csv", index=False)
    return summary, base, tq_edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)                  # never reuse a directory
    t0 = time.time()
    cfg = load_config()
    s = cfg.subtype
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if digest(panel_path, "md5") != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    B = 200 if a.smoke else 4000
    config = {"plan": str(PLAN.relative_to(ROOT)),
              "sha256": {f: digest(ROOT / f) for f in [str(PLAN.relative_to(ROOT))] + CODE},
              "panel_md5": PANEL_MD5, "seed": SEED, "bootstrap": B, "smoke": a.smoke,
              "smoke_subset": "first 150 patients by sorted PATNO in each cohort" if a.smoke else None}
    (out / "c1_config.json").write_text(json.dumps(config, indent=2))
    log("config written")

    part2, part3 = loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg)
    table = RS.item_visit_table(part2, part3, s)
    panel = pd.read_csv(panel_path, low_memory=False)
    if a.smoke:
        panel = panel[panel.PATNO.isin(np.sort(panel.PATNO.unique())[:150])]
    frame = RS.build_visit_frame(panel, table, s)
    valid = RS.validate_reconstruction(frame)
    if not valid["exact"]:
        print(f"reconstruction mismatch: {valid}", file=sys.stderr)
        return 3
    log(f"reconstruction exact: {valid}")

    # ---- item-level missingness within the cohort ---------------------------------
    items = table.merge(panel[["PATNO", "EVENT_ID"]], on=["PATNO", "EVENT_ID"])
    raw = items[list(s.all_items)].apply(pd.to_numeric, errors="coerce")
    pd.DataFrame({"item": list(s.all_items),
                  "group": ["tremor" if c in s.tremor_items else "pigd" for c in s.all_items],
                  "part": ["II" if c in RS.PART2_ITEMS else "III" for c in s.all_items],
                  "absent": [int(raw[c].isna().sum()) for c in s.all_items],
                  "code_101": [int((raw[c] == 101).sum()) for c in s.all_items],
                  "out_of_range_other": [int((raw[c].notna() & (raw[c] != 101) &
                                              ((raw[c] < 0) | (raw[c] > 4))).sum())
                                         for c in s.all_items],
                  "visits_with_item_rows": len(items)}).to_csv(out / "item_missingness.csv",
                                                               index=False)

    # ---- primary cohort -----------------------------------------------------------------
    summ, base, tq_edges = run_cohort("primary", frame, B, SEED, out)
    yearly_tab = yearly(frame, B, SEED)
    yearly_tab.to_csv(out / "primary_F3_yearly_means.csv", index=False)
    pairs = consecutive_changes(frame)
    pairs.groupby("kind").agg(pairs=("PATNO", "size"), patients=("PATNO", "nunique"),
                              d_tremor_median=("d_tremor", "median"),
                              d_pigd_median=("d_pigd", "median"),
                              abs_total_median=("d_tremor", lambda x: np.median(
                                  np.abs(x) + np.abs(pairs.loc[x.index, "d_pigd"]))),
                              share_abs_total_le2=("d_tremor", lambda x: np.mean(
                                  (np.abs(x) + np.abs(pairs.loc[x.index, "d_pigd"])) <= 2))) \
        .to_csv(out / "primary_consecutive_changes.csv")
    lab = frame[frame.subtype_name.notna()]
    ch = lab.groupby("PATNO").subtype_name.nunique()
    first_last = lab.sort_values("years_from_baseline").groupby("PATNO").agg(
        t0=("years_from_baseline", "first"), t1=("years_from_baseline", "last"),
        tr0=("tremor_sum", "first"), tr1=("tremor_sum", "last"),
        pg0=("pigd_sum", "first"), pg1=("pigd_sum", "last"), n=("PATNO", "size"))
    first_last = first_last[first_last.n >= 2]
    first_last["changer"] = ch.loc[first_last.index] > 1
    fu = (first_last.t1 - first_last.t0).replace(0, np.nan)
    first_last.assign(tremor_per_yr=(first_last.tr1 - first_last.tr0) / fu,
                      pigd_per_yr=(first_last.pg1 - first_last.pg0) / fu) \
        .groupby("changer")[["n", "tr0", "pg0", "tremor_per_yr", "pigd_per_yr"]] \
        .median().assign(patients=first_last.groupby("changer").size()) \
        .to_csv(out / "primary_changers_vs_nonchangers_medians.csv")
    tl = lab.sort_values(["PATNO", "years_from_baseline"])
    gaps = tl.groupby("PATNO").years_from_baseline.diff().dropna()
    fol = tl.groupby("PATNO").years_from_baseline.agg(lambda x: x.max() - x.min())
    fol = fol[tl.groupby("PATNO").size() >= 2]
    summ["interval_quantiles"] = gaps.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
    summ["followup_quantiles"] = fol.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
    summ["tremor_quintile_edges"] = [float(x) for x in tq_edges]

    # ---- examples (illustration only) -----------------------------------------------------
    seqf = base[base.n_labelled >= 2].reset_index()[["PATNO", "baseline_label", "n_labelled"]] \
        .rename(columns={"n_labelled": "n_visits"})
    examples, ex_log, ex_edges, strata = RS.example_patients(seqf, 16, SEED)
    examples.to_csv(out / "example_patients.csv", index=False)
    strata.to_csv(out / "example_strata.csv", index=False)

    # ---- sensitivity constructions --------------------------------------------------------
    def agree(col):
        both = frame[frame.subtype_name.notna() & frame[col].notna()]
        return {"crosstab": pd.crosstab(both.subtype_name, both[col]).to_dict(),
                "agreement": float((both.subtype_name == both[col]).mean()),
                "visits_both": int(len(both)),
                "labelled_primary_only": int((frame.subtype_name.notna() & frame[col].isna()).sum()),
                "labelled_sensitivity_only": int((frame.subtype_name.isna() & frame[col].notna()).sum())}
    sens = {"part3_only": agree("subtype_name_p3"), "one_missing_pigd_item": agree("subtype_name_miss1"),
            "inner_band": agree("subtype_name_inner")}
    gained = frame[frame.subtype_name.isna() & frame.subtype_name_miss1.notna()]
    sens["one_missing_pigd_item"]["gained_visits_by_reason"] = gained.reason.value_counts().to_dict()
    sens["one_missing_pigd_item"]["gained_visits_label"] = gained.subtype_name_miss1.value_counts().to_dict()
    sens["one_missing_pigd_item"]["gained_visits_pigd_101"] = int((gained.pigd_n_101 > 0).sum())
    sens["one_missing_pigd_item"]["gained_visits_pigd_absent"] = int((gained.pigd_n_absent > 0).sum())
    sens["one_missing_pigd_item"]["assumption"] = (
        "PIGD = mean of the 4 observed items: the missing item is implicitly set to the "
        "mean of the patient's other PIGD items at that visit; if the item was unrateable "
        "because of severity (e.g. postural stability), this understates PIGD")

    # ---- OFF/ON same-visit agreement (descriptive, NOT test-retest) -----------------------
    pairs_oo = RS.off_on_pairs(part2, part3, s)
    pairs_oo = pairs_oo.merge(panel[["PATNO", "EVENT_ID"]], on=["PATNO", "EVENT_ID"])
    oo = {"paired_visits": int(len(pairs_oo)), "patients": int(pairs_oo.PATNO.nunique())}
    for tag, cols in (("full", ("subtype_name_OFF", "subtype_name_ON")),
                      ("part3_only", ("subtype_name_p3_OFF", "subtype_name_p3_ON"))):
        d = pairs_oo.dropna(subset=list(cols))
        oo[tag] = {"pairs_with_both_labels": int(len(d)), "patients": int(d.PATNO.nunique()),
                   "agreement": float((d[cols[0]] == d[cols[1]]).mean()) if len(d) else None,
                   "crosstab_OFF_rows_ON_cols": pd.crosstab(d[cols[0]], d[cols[1]]).to_dict()}
    dd = pairs_oo.dropna(subset=["tremor_sum_OFF", "tremor_sum_ON", "pigd_sum_OFF", "pigd_sum_ON"])
    oo["ON_minus_OFF_tremor_sum_quantiles"] = (dd.tremor_sum_ON - dd.tremor_sum_OFF).quantile(
        [.1, .25, .5, .75, .9]).to_dict()
    oo["ON_minus_OFF_pigd_sum_quantiles"] = (dd.pigd_sum_ON - dd.pigd_sum_OFF).quantile(
        [.1, .25, .5, .75, .9]).to_dict()
    if "HRPOSTMED_ON" in pairs_oo:
        oo["HRPOSTMED_ON_available_share"] = float(pairs_oo.HRPOSTMED_ON.notna().mean())
    oo["note"] = ("Same-visit OFF vs ON exams: medication response, time-of-day, order and "
                  "rater effects combined. Not test-retest reliability. Part II items are "
                  "shared by both exams; the Part III-only row removes that shared component.")
    pairs_oo.to_csv(out / "off_on_pairs.csv.gz", index=False)

    # ---- genetic-PD cohort (descriptive) --------------------------------------------------
    gen_panel = RS.genetic_pd_panel(cfg, RS.subscores_from_items(table, s))
    if a.smoke:
        gen_panel = gen_panel[gen_panel.PATNO.isin(np.sort(gen_panel.PATNO.unique())[:150])]
    gen_frame = RS.build_visit_frame(gen_panel, table, s)
    gsum, _, _ = run_cohort("genetic_pd", gen_frame, B, SEED, out, tq_edges=tq_edges)
    gsum["subgroups"] = gen_panel.drop_duplicates("PATNO").subgroup.value_counts().to_dict()
    gsum["cohort_values"] = gen_panel.COHORT.value_counts().to_dict()
    gen_frame.to_csv(out / "genetic_pd_visits.csv.gz", index=False)

    figures(out, frame, yearly_tab, examples, pairs, tq_edges)
    frame.to_csv(out / "primary_visits.csv.gz", index=False)
    meta = {"reconstruction": valid, "primary": summ, "genetic_pd": gsum,
            "sensitivity_constructions": sens, "off_on": oo,
            "examples": {"selected": int(len(examples)), "log": ex_log,
                         "visit_count_tertile_edges": [float(x) for x in ex_edges],
                         "method": "baseline label x visit-count tertile; 2 per stratum for "
                                   "the largest strata, 1 for the rest, 16 total; seed 42; "
                                   "illustration only"},
            "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__},
            "host": platform.node(), "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"done in {meta['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
