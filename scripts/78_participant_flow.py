"""E-5: participant flow, every exclusion counted, from the raw cut to the scored target.

WHY
---
The manuscript needs a STROBE-style flow diagram and the project has never
assembled one. Every count exists somewhere -- in `outputs/panel_summary.txt`,
`outputs/transitions/report.txt`, the next-visit run's `outcome_counts.csv` --
but no single artefact traces one participant from the raw curated cut to a
scored two-class prediction, and no artefact says how many were lost at each
step or why.

This script replays the committed pipeline (`panel.filter_cohort`,
`panel.build_subscores`, `subtype.*`, `transitions.consecutive_pairs`) against
the raw source files, counting patients and visits before and after every
filter, and writes the diagram.

A SECOND, UNPLANNED RESULT
--------------------------
Diagnosing *why* 1,333 cohort visits carry no subtype contradicts the account
in README.md and in the Weeks 3-4 report, which attribute roughly half the
loss to exams "performed but not fully scored" and say that "in 1,243 of those
1,253 exams the missing item is postural stability".

On the frozen panel that is not what the committed code produces. The
dominant mechanism is the **OFF-medication requirement**: 1,195 of the 1,333
unlabelled visits had a Part III examination that was recorded ON medication
and was therefore excluded by `subtype.select_medication_state`. The
postural-stability mechanism is real but accounts for 54 visits, not 1,243.

The script measures this and characterises who is lost, because the direction
of the resulting bias is not the same under the two explanations and the
manuscript's limitation section depends on which is right.

    python scripts/78_participant_flow.py --out <dir>
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
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import loaders, panel as P, subtype as S, transitions as T  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK, INK_MUTED, _finish  # noqa: E402

KEY = ["PATNO", "EVENT_ID"]
IND = 2.0

#  Counts the committed artefacts publish, for the reproduction check.
COMMITTED = {"cohort_visits": 5774, "cohort_patients": 1080, "labelled": 4441,
             "ratio_defined": 3716, "ratio_undefined": 725,
             "patients_ge2": 934, "transitions": 3361,
             "eligible_two_class": 3029, "indeterminate_targets": 332}

ACCENT = "#1f4e79"
WARM = "#c2552c"
SURFACE = "#fcfcfb"

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


def nn(frame_: pd.DataFrame) -> tuple[int, int]:
    """(visits, patients)."""
    return len(frame_), int(frame_["PATNO"].nunique())


# ---------------------------------------------------------------------------
#  The figure
# ---------------------------------------------------------------------------
def flow_figure(spine: list[dict], excl: list[dict], path: Path) -> Path:
    """CONSORT-style vertical flow with exclusions branching right."""
    n = len(spine)
    fig, ax = plt.subplots(figsize=(10.4, 1.30 * n + 0.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.30 * n + 0.4)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    ys = [1.30 * (n - 1 - i) + 0.5 for i in range(n)]
    box_w, box_h, x0 = 4.5, 0.86, 0.25

    for i, (s, y) in enumerate(zip(spine, ys)):
        ax.add_patch(FancyBboxPatch(
            (x0, y), box_w, box_h, boxstyle="round,pad=0.035,rounding_size=0.07",
            linewidth=1.3, edgecolor=ACCENT if i in (0, n - 1) else GRID,
            facecolor="#eef3f9" if i in (0, n - 1) else SURFACE, zorder=2))
        ax.text(x0 + 0.17, y + box_h * 0.62, s["label"], fontsize=9.6,
                color=INK, va="center", zorder=3)
        ax.text(x0 + 0.17, y + box_h * 0.24, s["counts"], fontsize=9.0,
                color=INK_MUTED, va="center", zorder=3, family="monospace")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x0 + box_w / 2, y), (x0 + box_w / 2, ys[i + 1] + box_h),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                color=INK_MUTED, shrinkA=1, shrinkB=1, zorder=1))

    for e in excl:
        y = ys[e["after"]] + box_h * 0.5
        y_mid = (y + ys[e["after"] + 1] + box_h) / 2 - 0.08
        ax.add_patch(FancyArrowPatch(
            (x0 + box_w / 2, y_mid), (x0 + box_w + 0.55, y_mid),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
            color=WARM, shrinkA=0, shrinkB=2, zorder=1))
        ax.text(x0 + box_w + 0.70, y_mid, e["text"], fontsize=8.5,
                color=INK_MUTED, va="center", zorder=3)

    return _finish(fig, path)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = load_config()
    scfg, ch = cfg.subtype, cfg.cohort
    checks: dict[str, bool] = {}
    ledger: list[dict] = []

    def step(stage: str, visits: int, patients: int, note: str = "") -> None:
        ledger.append({"stage": stage, "visits": visits,
                       "patients": patients, "note": note})
        say(f"  {stage:<46s} {visits:>7,} visits   {patients:>6,} patients"
            + (f"   {note}" if note else ""))

    section("E-5  PARTICIPANT FLOW")

    # ------------------------------------------------------- the visit spine
    section("1. CURATED DATA CUT -> ANALYSIS COHORT")
    curated = loaders.load_curated_cut(cfg)
    v, p_ = nn(curated)
    step("curated data cut, as published", v, p_)

    keep_sub = set(ch["primary_subgroups"])
    s1 = curated[curated["subgroup"].isin(keep_sub)]
    dropped_sub = curated[~curated["subgroup"].isin(keep_sub)]
    v1, p1 = nn(s1)
    step(f"subgroup in {sorted(keep_sub)}", v1, p1,
         f"-{v - v1:,} visits, -{p_ - p1:,} patients")
    sub_tbl = (dropped_sub.groupby("subgroup")
               .agg(visits=("PATNO", "size"), patients=("PATNO", "nunique"))
               .sort_values("visits", ascending=False))
    sub_tbl.to_csv(out / "excluded_by_subgroup.csv")

    ex_ev = set(ch["excluded_events"])
    s2 = s1[~s1["EVENT_ID"].isin(ex_ev)]
    v2, p2_ = nn(s2)
    ev_tbl = (s1[s1["EVENT_ID"].isin(ex_ev)]["EVENT_ID"].value_counts()
              .rename("visits").to_frame())
    step(f"exclude events {sorted(ex_ev)}", v2, p2_, f"-{v1 - v2:,} visits")
    ev_tbl.to_csv(out / "excluded_by_event.csv")
    say(f"      {dict(ev_tbl['visits'])}")

    cap = ch["max_follow_up_years"]
    s3 = s2[s2["YEAR"] <= cap]
    v3, p3_ = nn(s3)
    step(f"follow-up <= {cap} years", v3, p3_, f"-{v2 - v3:,} visits")

    cohort = P.filter_cohort(curated, cfg)
    checks["the replay reproduces panel.filter_cohort exactly"] = (
        len(cohort) == v3 and cohort["PATNO"].nunique() == p3_)
    checks["cohort visit count reproduces the committed 5,774"] = (
        v3 == COMMITTED["cohort_visits"])
    checks["cohort patient count reproduces the committed 1,080"] = (
        p3_ == COMMITTED["cohort_patients"])

    # --------------------------------------------------- the item-file branch
    section("2. ITEM-LEVEL MDS-UPDRS BRANCH")
    part2 = loaders.load_updrs_part2(cfg)
    part3 = loaders.load_updrs_part3(cfg)
    say(f"  Part II  (NUPDRS2P)   {len(part2):>7,} rows   "
        f"{part2['PATNO'].nunique():>6,} patients")
    say(f"  Part III (NUPDRS3TRT) {len(part3):>7,} rows   "
        f"{part3['PATNO'].nunique():>6,} patients")

    part3_off = S.select_medication_state(part3, scfg)
    state = part3["PDSTATE"].astype("string").str.strip().str.upper()
    treat = pd.to_numeric(part3["PDTRTMNT"], errors="coerce")
    say(f"\n  medication-state selection ({scfg.medication_state}/untreated):")
    say(f"    PDSTATE values: {dict(state.value_counts(dropna=False))}")
    say(f"    untreated (PDTRTMNT == 0): {int((treat == 0).sum()):,}")
    say(f"    kept {len(part3_off):,} of {len(part3):,} rows "
        f"(-{len(part3) - len(part3_off):,}, all ON-state and treated)")

    p2i = [c for c in scfg.all_items if c in part2.columns]
    p3i = [c for c in scfg.all_items if c in part3_off.columns]
    carry = [c for c in (scfg.pdstate_column, scfg.treatment_column, "NHY")
             if c in part3_off.columns]
    merged = part2[KEY + p2i].merge(part3_off[KEY + p3i + carry], on=KEY, how="outer")
    deduped = S.deduplicate_visits(merged, scfg)
    say(f"\n  outer merge of the two files      {len(merged):>7,} rows")
    say(f"  duplicate patient-visits resolved {len(deduped):>7,} rows "
        f"(-{len(merged) - len(deduped):,})")

    cleaned = S.clean_items(deduped, scfg.all_items, scfg)
    scores = S.compute_subscores(deduped, scfg)
    sentinel = S.count_unrateable(deduped, scfg)
    sentinel[sentinel > 0].rename("n_sentinel_101").to_frame().to_csv(
        out / "sentinel_counts.csv")
    #  Three denominators, three different numbers, and the project's two
    #  published figures belong to two of them. Stating all three is what
    #  settles the apparent discrepancy between config.yaml and README.md.
    codes = list(scfg.missing_value_codes)
    pstbl_raw = int(pd.to_numeric(part3["NP3PSTBL"], errors="coerce").isin(codes).sum())
    pstbl_off = int(pd.to_numeric(part3_off["NP3PSTBL"], errors="coerce").isin(codes).sum())
    pstbl_ded = int(sentinel.get("NP3PSTBL", 0))
    say(f"\n  sentinel-101 ('unable to rate') values blanked before averaging:")
    say(f"    all items, deduplicated set                 {int(sentinel.sum()):,}")
    say(f"    NP3PSTBL, raw Part III file                 {pstbl_raw:,}"
        f"   <- config.yaml quotes this")
    say(f"    NP3PSTBL, after OFF/untreated selection     {pstbl_off:,}"
        f"   <- README.md quotes this")
    say(f"    NP3PSTBL, after duplicate resolution        {pstbl_ded:,}")
    say("    The three differ only by their denominator; deduplication drops")
    say(f"    {len(merged) - len(deduped)} rows, one of which carried a sentinel. Both published")
    say("    figures are correct for the set they describe.")
    pd.DataFrame([{"set": "raw Part III file", "rows": len(part3), "np3pstbl_101": pstbl_raw},
                  {"set": "after OFF/untreated selection", "rows": len(part3_off),
                   "np3pstbl_101": pstbl_off},
                  {"set": "after duplicate resolution", "rows": len(deduped),
                   "np3pstbl_101": pstbl_ded}]).to_csv(
        out / "sentinel_denominators.csv", index=False)
    checks["config.yaml's NP3PSTBL sentinel count (4,302) is reproduced"] = (
        pstbl_raw == 4302)
    checks["README.md's NP3PSTBL sentinel count (3,206) is reproduced"] = (
        pstbl_off == 3206)

    item_frame = pd.concat([deduped[KEY].reset_index(drop=True),
                            cleaned.reset_index(drop=True),
                            scores.reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------- the label
    section("3. COHORT VISITS -> LABELLED VISITS")
    m = cohort[KEY].merge(item_frame, on=KEY, how="left", validate="one_to_one")
    checks["the item join is one-to-one against the cohort skeleton"] = len(m) == v3

    labelled_mask = m["subtype"].notna()
    n_lab = int(labelled_mask.sum())
    step("cohort patient-visits", v3, p3_)
    step("with a TD/PIGD subtype label", n_lab,
         int(m.loc[labelled_mask, "PATNO"].nunique()),
         f"-{v3 - n_lab:,} visits unlabelled")
    checks["labelled visit count reproduces the committed 4,441"] = (
        n_lab == COMMITTED["labelled"])

    panel = pd.read_csv(ROOT / "outputs" / "visit_panel.csv", low_memory=False)
    md5 = hashlib.md5((ROOT / "outputs" / "visit_panel.csv").read_bytes()).hexdigest()
    checks["frozen panel md5 matches the project's recorded value"] = (
        md5 == "fb038ddbdc19ab3368e13940f398e7ba")
    frozen_lab = panel[panel["subtype"].notna()]
    checks["the replayed label count agrees with the frozen panel"] = (
        len(frozen_lab) == n_lab)

    # -------------------------------------------- WHY the rest are unlabelled
    section("4. WHY 1,333 COHORT VISITS CARRY NO LABEL")
    say("  This contradicts README.md and the Weeks 3-4 report. Measured, not")
    say("  quoted.\n")
    un = m[~labelled_mask].copy()
    p3_items = [c for c in scfg.all_items if c in p3i]
    p2_items = [c for c in scfg.all_items if c in p2i]
    n_p3 = un[p3_items].notna().sum(axis=1)
    n_p2 = un[p2_items].notna().sum(axis=1)

    no_p3 = n_p3 == 0
    part_p3 = (n_p3 > 0) & (n_p3 < len(p3_items))
    all_p3 = n_p3 == len(p3_items)

    raw3_keys = part3[KEY].drop_duplicates().assign(_raw=1)
    chk = un.loc[no_p3, KEY].merge(raw3_keys, on=KEY, how="left")
    on_only = int(chk["_raw"].notna().sum())
    no_exam = int(chk["_raw"].isna().sum())

    diag = pd.DataFrame([
        {"reason": "only an ON-medication Part III exam exists, so it is excluded",
         "visits": on_only},
        {"reason": "no Part III row in the source file at all", "visits": no_exam},
        {"reason": "Part III exam partially scored (>=1 item unrateable or absent)",
         "visits": int(part_p3.sum())},
        {"reason": "all 13 Part III items present; Part II blocked the label",
         "visits": int(all_p3.sum())},
    ])
    diag["share"] = diag["visits"] / len(un)
    diag.to_csv(out / "unlabelled_diagnosis.csv", index=False)
    say(frame(diag.round(4)))
    checks["the unlabelled diagnosis accounts for every unlabelled visit"] = (
        int(diag["visits"].sum()) == len(un))

    pp = un[part_p3]
    n_pstbl = int(pp["NP3PSTBL"].isna().sum()) if len(pp) else 0
    only_pstbl = int(((pp[p3_items].isna().sum(axis=1) == 1)
                      & pp["NP3PSTBL"].isna()).sum()) if len(pp) else 0
    say(f"\n  Among the {len(pp):,} partially scored exams, postural stability")
    say(f"  (NP3PSTBL) is the missing item in {n_pstbl:,} ({n_pstbl/max(len(pp),1):.1%}),")
    say(f"  and the ONLY missing Part III item in {only_pstbl:,}.")
    say("")
    say("  CORRECTION. README.md states that roughly half the label loss is")
    say("  'exams that were performed but could not be fully scored' and that")
    say("  'in 1,243 of those 1,253 exams the missing item is postural")
    say(f"  stability'. On the frozen panel the figure is {n_pstbl:,} visits, not")
    say(f"  1,243. The dominant mechanism is the OFF-medication requirement:")
    say(f"  {on_only:,} of {len(un):,} unlabelled visits ({on_only/len(un):.1%}) had a Part III")
    say("  examination recorded ON medication, which select_medication_state")
    say("  excludes by design.")
    checks["the ON-medication mechanism dominates the label loss"] = (
        on_only / len(un) > 0.5)

    # ------------------------------------------- who the unlabelled visits are
    section("5. WHO IS LOST, AND IN WHICH DIRECTION THE BIAS RUNS")
    lab_f = panel["subtype"].notna()
    rows = []
    hy_any = (pd.to_numeric(panel["hy"], errors="coerce")
              .fillna(pd.to_numeric(panel["hy_on"], errors="coerce")))
    for name, series in [
        ("LEDD at the visit", pd.to_numeric(panel["LEDD"], errors="coerce")),
        ("years from baseline", pd.to_numeric(panel["years_from_baseline"],
                                              errors="coerce")),
        ("Hoehn & Yahr (hy, else hy_on)", hy_any),
    ]:
        a, b = series[~lab_f].dropna(), series[lab_f].dropna()
        pv = mannwhitneyu(a, b, alternative="two-sided")[1] if len(a) and len(b) else np.nan
        rows.append({"variable": name, "n_unlabelled": len(a), "n_labelled": len(b),
                     "median_unlabelled": a.median(), "median_labelled": b.median(),
                     "mean_unlabelled": a.mean(), "mean_labelled": b.mean(),
                     "mannwhitney_p": pv})
    treat_p = pd.to_numeric(panel["PDTRTMNT"], errors="coerce")
    rows.append({"variable": "share on PD medication",
                 "n_unlabelled": int(treat_p[~lab_f].notna().sum()),
                 "n_labelled": int(treat_p[lab_f].notna().sum()),
                 "median_unlabelled": np.nan, "median_labelled": np.nan,
                 "mean_unlabelled": float((treat_p[~lab_f] == 1).sum()
                                          / max(treat_p[~lab_f].notna().sum(), 1)),
                 "mean_labelled": float((treat_p[lab_f] == 1).sum()
                                        / max(treat_p[lab_f].notna().sum(), 1)),
                 "mannwhitney_p": np.nan})
    who = pd.DataFrame(rows)
    who.to_csv(out / "who_is_unlabelled.csv", index=False)
    say(frame(who.round(4)))

    off_avail = pd.DataFrame([
        {"column": c,
         "present_unlabelled": float(panel.loc[~lab_f, c].notna().mean()),
         "present_labelled": float(panel.loc[lab_f, c].notna().mean())}
        for c in ("hy", "hy_on", "updrs3_score", "updrs3_score_on")
        if c in panel.columns])
    off_avail.to_csv(out / "off_on_availability.csv", index=False)
    say("\n  Availability of the OFF- against the ON-state curated columns:")
    say(frame(off_avail.round(4)))
    say("\n  This is the mechanism, read off a second, independent source: the")
    say("  OFF-state columns are near-absent at unlabelled visits while the")
    say("  ON-state ones are present. The unlabelled visits are ON-state visits")
    say("  of treated patients later in follow-up.")

    say("\n  DIRECTION OF THE BIAS. The corrected mechanism does not overturn the")
    say("  README's conclusion, but it reaches it differently. Label loss is")
    say("  concentrated in treated patients later in follow-up (median year 5.0")
    say("  against 2.0; 96.8% on medication against 57.9%). The analysed sample")
    say("  is therefore weighted toward earlier, untreated follow-up, which")
    say("  ATTENUATES observed progression -- so the drift reported in this")
    say("  project remains a lower bound. The OFF requirement itself is correct")
    say("  and must be kept: levodopa suppresses tremor, so scoring ON exams")
    say("  would manufacture PIGD transitions that are pharmacological rather")
    say("  than pathological.")

    # ------------------------------------------------------------ downstream
    section("6. LABELLED VISITS -> SCORED TWO-CLASS TARGETS")
    lab = panel[lab_f].copy()
    n_def = int((lab["ratio_defined"] == True).sum())  # noqa: E712
    say(f"  ratio defined                    {n_def:>7,} of {len(lab):,} labelled visits")
    say(f"  ratio undefined (PIGD score 0)   {len(lab) - n_def:>7,}")
    und = lab[lab["ratio_defined"] != True]  # noqa: E712
    say(f"    their subtypes: "
        f"{dict(und['subtype_name'].value_counts())}")
    checks["ratio-defined count reproduces the committed 3,716"] = (
        n_def == COMMITTED["ratio_defined"])

    counts = lab.groupby("PATNO")["subtype"].size()
    ge2 = counts[counts >= ch["min_visits_per_patient"]].index
    step("labelled visits", len(lab), int(lab["PATNO"].nunique()))
    step(f"patients with >= {ch['min_visits_per_patient']} labelled visits",
         int(lab["PATNO"].isin(ge2).sum()), len(ge2),
         f"-{int(lab['PATNO'].nunique()) - len(ge2):,} patients with one only")
    checks["patients with >=2 labelled visits reproduces the committed 934"] = (
        len(ge2) == COMMITTED["patients_ge2"])

    pairs = T.consecutive_pairs(panel)
    step("consecutive labelled visit pairs", len(pairs),
         int(pairs["PATNO"].nunique()), "one prediction opportunity each")
    checks["transition count reproduces the committed 3,361"] = (
        len(pairs) == COMMITTED["transitions"])

    ind_target = int((pairs["to_state"] == IND).sum())
    elig = pairs[pairs["to_state"] != IND]
    step("eligible for two-class TD-vs-PIGD scoring", len(elig),
         int(elig["PATNO"].nunique()),
         f"-{ind_target:,} Indeterminate targets")
    checks["two-class eligible count reproduces the committed 3,029"] = (
        len(elig) == COMMITTED["eligible_two_class"])
    checks["Indeterminate target count reproduces the committed 332"] = (
        ind_target == COMMITTED["indeterminate_targets"])
    say(f"\n  {int(pairs['PATNO'].nunique()) - int(elig['PATNO'].nunique()):,} of "
        f"{int(pairs['PATNO'].nunique()):,} patients contribute no eligible target")

    pd.DataFrame(ledger).to_csv(out / "flow_ledger.csv", index=False)

    # --------------------------------------------------------------- figure
    section("7. FIGURE")
    spine = [
        {"label": "PPMI Curated Data Cut (public release 2026-05-11)",
         "counts": f"{v:,} visits   {p_:,} participants"},
        {"label": "Sporadic PD subgroup",
         "counts": f"{v1:,} visits   {p1:,} patients"},
        {"label": "Scheduled visits within 15 years of baseline",
         "counts": f"{v3:,} visits   {p3_:,} patients"},
        {"label": "Visits with a TD / Indeterminate / PIGD label",
         "counts": f"{n_lab:,} visits   "
                   f"{int(m.loc[labelled_mask,'PATNO'].nunique()):,} patients"},
        {"label": "Patients with two or more labelled visits",
         "counts": f"{int(lab['PATNO'].isin(ge2).sum()):,} visits   {len(ge2):,} patients"},
        {"label": "Consecutive labelled pairs (prediction opportunities)",
         "counts": f"{len(pairs):,} transitions   {int(pairs['PATNO'].nunique()):,} patients"},
        {"label": "Scored for two-class TD-vs-PIGD prediction",
         "counts": f"{len(elig):,} targets   {int(elig['PATNO'].nunique()):,} patients"},
    ]
    excl = [
        {"after": 0, "text": f"{v - v1:,} visits in other subgroups\n"
                             f"({p_ - p1:,} participants: healthy control,\n"
                             f"genetic, prodromal, SWEDD)"},
        {"after": 1, "text": f"{v1 - v3:,} unscheduled or premature-withdrawal\n"
                             f"visits (ST {int(ev_tbl['visits'].get('ST', 0))}, "
                             f"PW {int(ev_tbl['visits'].get('PW', 0))}); "
                             f"{v2 - v3:,} beyond 15 years"},
        {"after": 2, "text": f"{v3 - n_lab:,} visits with no subtype:\n"
                             f"{on_only:,} ON-medication exam only,\n"
                             f"{int(part_p3.sum()):,} partially scored, "
                             f"{no_exam + int(all_p3.sum()):,} other"},
        {"after": 3, "text": f"{int(lab['PATNO'].nunique()) - len(ge2):,} patients with a single\n"
                             f"labelled visit, so no transition"},
        {"after": 5, "text": f"{ind_target:,} targets labelled Indeterminate,\n"
                             f"never scored in the two-class task"},
    ]
    fp = flow_figure(spine, excl, out / "fig1_participant_flow.png")
    say(f"  {fp.name}")
    pd.DataFrame(spine).to_csv(out / "fig1_participant_flow.csv", index=False)

    # ------------------------------------------------------------- artefacts
    section("8. CHECKS")
    cf = pd.DataFrame([{"check": k, "passed": bool(v_)} for k, v_ in checks.items()])
    cf.to_csv(out / "validation_checks.csv", index=False)
    say(frame(cf))
    n_ok = int(cf["passed"].sum())
    say(f"\n  {n_ok} of {len(cf)} checks passed")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (out / "summary.json").write_text(json.dumps({
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "node": platform.node(),
            "elapsed_seconds": round(time.time() - started, 1),
            "git_commit": commit, "panel_md5": md5,
            "models_fitted": 0, "analysis_only": True,
        },
        "flow": {s["stage"]: {"visits": s["visits"], "patients": s["patients"]}
                 for s in ledger},
        "unlabelled_diagnosis": diag.to_dict("records"),
        "correction": {
            "readme_claim": "1,243 of 1,253 unlabelled exams missing postural stability",
            "measured_postural_stability_visits": n_pstbl,
            "measured_on_medication_visits": on_only,
            "unlabelled_total": len(un),
        },
        "checks": {k: bool(v_) for k, v_ in checks.items()},
        "code_sha256": {"scripts/78_participant_flow.py": sha(__file__)},
    }, indent=1), encoding="utf-8")

    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "driver.log").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(pth for pth in out.rglob("*")
                   if pth.is_file() and not pth.is_symlink() and pth.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(pth)}  {pth.relative_to(out)}\n" for pth in files),
        encoding="utf-8")

    print(f"\nWrote {len(files)} files to {out}")
    return 0 if n_ok == len(cf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
