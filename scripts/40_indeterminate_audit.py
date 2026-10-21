"""Part C2: Indeterminate-class audit (reversion difference and temporal clustering).

Plan: docs/cluster_phase/10_PART_C_RESEARCH_PLAN.md section 5 (approved). The
plan and code SHA-256 are written to c2_config.json BEFORE any statistic is
computed. Decision rule: plan Amendment 1 (2026-10-16), which supersedes the
original section 5.5 conditions; the superseded rule is reported for the record only. The reading rule is applied to the primary cohort and primary label
only; every sensitivity is reported beside it and never replaces it. Readings
describe observed label sequences; they are not evidence of a biological
state or of latent disease progression.

    python scripts/40_indeterminate_audit.py --out <new dir> [--smoke]

--smoke: first 150 patients (sorted PATNO) of each cohort, 200 bootstrap replicates.
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

from ppmi_subtype import indeterminate_audit as IA  # noqa: E402
from ppmi_subtype import loaders  # noqa: E402
from ppmi_subtype import raw_scores as RS  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.transitions import kaplan_meier  # noqa: E402

SEED = 42
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
PLAN = "docs/cluster_phase/10_PART_C_RESEARCH_PLAN.md"
PLAN_SHA_REV2 = "0eacdac5ac1f18da9621b67b6fb1a4ca46a1e6c833f600890930e4efeb009f40"
CODE = ["src/ppmi_subtype/indeterminate_audit.py", "src/ppmi_subtype/raw_scores.py",
        "scripts/40_indeterminate_audit.py", "src/ppmi_subtype/subtype.py",
        "src/ppmi_subtype/panel.py", "src/ppmi_subtype/transitions.py"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def analyse(frame, label_col, B, segment_kw=None, apply_rule=True):
    C, pats = IA.count_matrix(frame, label_col, segment_kw)
    tot = dict(zip(IA.COUNT_KEYS, C.sum(axis=0)))
    point, ci, undef = IA.bootstrap(C, B=B, seed=SEED)
    col = {k: i for i, k in enumerate(IA.COUNT_KEYS)}
    npos = lambda k: int((C[:, col[k]] > 0).sum())  # noqa: E731
    res = {
        "label": label_col, "segments": segment_kw or "none",
        "patients_in_frame": int(frame.PATNO.nunique()),
        "eligible_patients_ge2_labelled": int(tot["eligible"]),
        "entries": {"events": int(tot["entry_n"]), "observed_next": int(tot["entry_obs"]),
                    "unobserved": int(tot["entry_unobs"]), "outside_segment": int(tot["entry_out"]),
                    "reverted": int(tot["entry_rev"]), "sustained": int(tot["entry_sus"]),
                    "moved_on": int(tot["entry_mov"]),
                    "patients_with_event": npos("entry_n"),
                    "patients_with_observed_event": npos("entry_obs")},
        "direct": {"events": int(tot["direct_n"]), "observed_next": int(tot["direct_obs"]),
                   "unobserved": int(tot["direct_unobs"]), "outside_segment": int(tot["direct_out"]),
                   "reverted": int(tot["direct_rev"]), "sustained": int(tot["direct_sus"]),
                   "moved_on": int(tot["direct_mov"]),
                   "patients_with_event": npos("direct_n"),
                   "patients_with_observed_event": npos("direct_obs")},
        "exits": {"events": int(tot["exit_n"]), "observed_next": int(tot["exit_obs"]),
                  "back_to_Indeterminate": int(tot["exit_back"]), "unobserved": int(tot["exit_unobs"])},
        "two_step": {"entries_with_two_later_visits": int(tot["entry_obs2"]),
                     "sustained_two": int(tot["entry_sus2"])},
        "clustering": {L: {"O": float(tot[f"O_{L}"]), "E": float(tot[f"E_{L}"])} for L in IA.LABELS},
        "patients_with_ge2_Indeterminate_in_a_segment": int(tot["contrib_Indeterminate"]),
        "point": point, "ci95": ci, "undefined_replicates": undef,
        "wilson_pair_level": {"R_E": IA.wilson(tot["entry_rev"], tot["entry_obs"]),
                              "R_D": IA.wilson(tot["direct_rev"], tot["direct_obs"])},
        "E_Indeterminate": float(tot["E_Indeterminate"]),
        "clustering_evaluable": bool(tot["E_Indeterminate"] >= IA.E_MIN),
    }
    res["reading"] = (IA.reading(ci["delta_R"], ci["OE_Indeterminate"], tot["E_Indeterminate"])
                      if apply_rule else "not applied (descriptive sensitivity)")
    res["reading_if_rule_applied"] = IA.reading(ci["delta_R"], ci["OE_Indeterminate"],
                                                tot["E_Indeterminate"])
    res["superseded_original_rule_for_record"] = IA.reading_original(
        ci["delta_R"], ci["OE_Indeterminate"], tot["E_Indeterminate"])
    return res, C, pats


def annotate_events(frame):
    """Descriptive annotations for every primary event."""
    ev = IA.event_table(frame, "subtype_name")
    f = frame
    k, j = f.loc[ev.row_k], f.loc[ev.row_j]
    ev["dt"] = j.years_from_baseline.to_numpy() - k.years_from_baseline.to_numpy()
    ev["d_tremor"] = j.tremor_sum.to_numpy() - k.tremor_sum.to_numpy()
    ev["d_pigd"] = j.pigd_sum.to_numpy() - k.pigd_sum.to_numpy()
    ev["abs_total"] = ev.d_tremor.abs() + ev.d_pigd.abs()
    ev["margin_k"] = k.label_margin.to_numpy()
    ev["margin_j"] = j.label_margin.to_numpy()
    tk = pd.to_numeric(k.PDTRTMNT, errors="coerce").to_numpy()
    tj = pd.to_numeric(j.PDTRTMNT, errors="coerce").to_numpy()
    ev["treatment_change"] = np.where(np.isfinite(tk) & np.isfinite(tj), tk != tj, np.nan)
    ev["both_zero_at_j"] = ((j.tremor_sum == 0) & (j.pigd_sum == 0)).to_numpy()
    ev["pigd_zero_at_k"] = (k.pigd_sum == 0).to_numpy()
    # unlabelled panel visits skipped between k and j (availability gaps)
    sk = []
    for p, a, b in zip(ev.PATNO, k.years_from_baseline, j.years_from_baseline):
        g = f[(f.PATNO == p) & (f.years_from_baseline > a) & (f.years_from_baseline < b)]
        sk.append(int(g.subtype_name.isna().sum()))
    ev["unlabelled_visits_between"] = sk
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = load_config()
    s = cfg.subtype
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    B = 200 if a.smoke else 4000
    # ---- freeze: hashes before any computation ----------------------------------------
    config = {"plan": PLAN,
              "plan_version": "revision 2 + Amendment 1 (2026-10-16 IST / 2026-10-15T22:10Z)",
              "plan_sha256_pre_amendment": PLAN_SHA_REV2,
              "decision_rule": "Amendment 1: (i) delta_R lo > 0 and O/E hi < 1; (ii) delta_R hi "
                               "<= 0 and O/E lo > 1; (iii) otherwise, or E < 20",
              "plan_section": "5 (definitions) as amended by Amendment 1 (decision rule)",
              "sha256": {f: digest(ROOT / f) for f in [PLAN] + CODE},
              "panel_md5": digest(panel_path, "md5"), "seed": SEED, "bootstrap": B,
              "E_min": IA.E_MIN, "smoke": a.smoke,
              "smoke_subset": "first 150 patients by sorted PATNO per cohort" if a.smoke else None,
              "primary": "frozen-panel cohort, subtype_name (unchanged rule)",
              "sensitivities": ["part3_only", "one_missing_pigd_item", "inner_band",
                                "interval_le_1.5y (segments)", "treatment_stable (segments)",
                                "exclude_both_zero_visits", "genetic_pd (descriptive)"]}
    (out / "c2_config.json").write_text(json.dumps(config, indent=2))
    log("c2_config.json written (hashes recorded before computation)")
    if config["panel_md5"] != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2

    part2, part3 = loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg)
    table = RS.item_visit_table(part2, part3, s)
    panel = pd.read_csv(panel_path, low_memory=False)
    if a.smoke:
        panel = panel[panel.PATNO.isin(np.sort(panel.PATNO.unique())[:150])]
    frame = RS.build_visit_frame(panel, table, s)
    valid = RS.validate_reconstruction(frame)
    if not valid["exact"]:
        print(f"reconstruction mismatch {valid}", file=sys.stderr)
        return 3

    # ---- primary --------------------------------------------------------------------------
    primary, C, pats = analyse(frame, "subtype_name", B)
    log(f"primary: {primary['reading']}")

    # ---- sensitivities (never replace the primary) -------------------------------------
    fz = frame.copy()
    fz.loc[(fz.tremor_sum == 0) & (fz.pigd_sum == 0), "subtype_name"] = np.nan
    sens = {
        "part3_only": analyse(frame, "subtype_name_p3", B, apply_rule=False)[0],
        "one_missing_pigd_item": analyse(frame, "subtype_name_miss1", B, apply_rule=False)[0],
        "inner_band": analyse(frame, "subtype_name_inner", B, apply_rule=False)[0],
        "interval_le_1.5y": analyse(frame, "subtype_name", B, {"max_interval": 1.5},
                                    apply_rule=False)[0],
        "treatment_stable": analyse(frame, "subtype_name", B, {"treatment_stable": True},
                                    apply_rule=False)[0],
        "exclude_both_zero_visits": analyse(fz, "subtype_name", B, apply_rule=False)[0],
    }
    gen_panel = RS.genetic_pd_panel(cfg, RS.subscores_from_items(table, s))
    if a.smoke:
        gen_panel = gen_panel[gen_panel.PATNO.isin(np.sort(gen_panel.PATNO.unique())[:150])]
    gen_frame = RS.build_visit_frame(gen_panel, table, s)
    sens["genetic_pd"] = analyse(gen_frame, "subtype_name", B, apply_rule=False)[0]
    log("sensitivities done")

    # ---- descriptives -----------------------------------------------------------------------
    ev = annotate_events(frame)
    ev.to_csv(out / "primary_events.csv.gz", index=False)
    desc = ev.groupby("kind").agg(
        events=("PATNO", "size"), patients=("PATNO", "nunique"),
        dt_median=("dt", "median"), abs_total_median=("abs_total", "median"),
        share_abs_total_le2=("abs_total", lambda x: float((x <= 2).mean())),
        margin_j_eq1_share=("margin_j", lambda x: float((x == 1).mean())),
        treatment_change_share=("treatment_change", "mean"),
        both_zero_at_j=("both_zero_at_j", "sum"), pigd_zero_at_k=("pigd_zero_at_k", "sum"),
        with_unlabelled_visits_between=("unlabelled_visits_between", lambda x: int((x > 0).sum())))
    desc.to_csv(out / "T4_event_descriptives.csv")
    lab = frame[frame.subtype_name.notna()]
    lab.groupby("subtype_name").label_margin.value_counts().unstack(fill_value=0) \
        .to_csv(out / "label_margin_by_label.csv")
    km_rows = []
    for st in IA.LABELS:
        e = IA.exit_episodes(frame, "subtype_name", st)
        fe = e[e.followed]
        km = kaplan_meier(fe.time, fe.event).assign(state=st, patients=len(fe),
                                                    no_followup=int((~e.followed).sum()))
        km_rows.append(km)
    km = pd.concat(km_rows)
    km.to_csv(out / "F6_time_to_exit_km.csv", index=False)

    rows = []
    for name, r in [("PRIMARY", primary)] + list(sens.items()):
        rows.append({"analysis": name, "label": r["label"], "segments": str(r["segments"]),
                     "eligible_patients": r["eligible_patients_ge2_labelled"],
                     "entries": r["entries"]["events"], "entries_observed": r["entries"]["observed_next"],
                     "entries_unobserved": r["entries"]["unobserved"],
                     "entries_outside_segment": r["entries"]["outside_segment"],
                     "direct": r["direct"]["events"], "direct_observed": r["direct"]["observed_next"],
                     "direct_unobserved": r["direct"]["unobserved"],
                     "R_E": r["point"]["R_E"], "R_E_lo": r["ci95"]["R_E"][0], "R_E_hi": r["ci95"]["R_E"][1],
                     "R_D": r["point"]["R_D"], "R_D_lo": r["ci95"]["R_D"][0], "R_D_hi": r["ci95"]["R_D"][1],
                     "delta_R": r["point"]["delta_R"], "delta_R_lo": r["ci95"]["delta_R"][0],
                     "delta_R_hi": r["ci95"]["delta_R"][1],
                     "O_Ind": r["clustering"]["Indeterminate"]["O"],
                     "E_Ind": r["clustering"]["Indeterminate"]["E"],
                     "OE_Ind": r["point"]["OE_Indeterminate"],
                     "OE_Ind_lo": r["ci95"]["OE_Indeterminate"][0],
                     "OE_Ind_hi": r["ci95"]["OE_Indeterminate"][1],
                     "OE_TD": r["point"]["OE_TD"], "OE_PIGD": r["point"]["OE_PIGD"],
                     "clustering_evaluable": r["clustering_evaluable"],
                     "reading": r["reading"], "reading_if_rule_applied": r["reading_if_rule_applied"],
                     "superseded_original_rule_for_record": r["superseded_original_rule_for_record"]})
    pd.DataFrame(rows).to_csv(out / "T5_T7_statistics.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for st, c in (("TD", "C0"), ("Indeterminate", "C2"), ("PIGD", "C3")):
        k = km[km.state == st]
        ax.step(k.time, k.survival, where="post", color=c,
                label=f"{st} (n={int(k.patients.iloc[0])} patients)")
    ax.set_xlabel("years from first visit with the label")
    ax.set_ylabel("probability label not yet left")
    ax.set_title("F6. Time to first different label (Kaplan-Meier; censored at last "
                 "labelled visit)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"F6_time_to_exit.{ext}", dpi=150)
    plt.close(fig)

    meta = {"reconstruction": valid, "primary": primary, "sensitivities": sens,
            "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__},
            "host": platform.node(), "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"done in {meta['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
