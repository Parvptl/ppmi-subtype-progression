"""Stage 2 data audit: item and item-sum structure for an integer-score model.

Descriptive only (doc 16, Step 1). No model is fitted and nothing is simulated.
Reuses the Part C reconstruction (`raw_scores`), validated exactly against the
frozen panel, and the existing label rule (`subtype.classify` via the panel).

Cohort: the frozen sporadic-PD panel's labelled visits (full three-state label,
PIGD = 0 visits included), primary analysis set = patients with >= 2 labelled
visits. The D1 ratio-defined cohort (821 patients) is reported alongside.

    python scripts/44_stage2_data_audit.py --out experiments/<UTC>_stage2_data_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import loaders  # noqa: E402
from ppmi_subtype import raw_scores as RS  # noqa: E402
from ppmi_subtype import stage1_checks as S1C  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
LAB = {"TD": 0, "Indeterminate": 1, "PIGD": 2}


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def _keys(o):
    """Stringify tuple keys (from multi-key groupbys) so the results serialise."""
    if isinstance(o, dict):
        return {(" | ".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_keys(v) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    s = cfg.subtype
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    conf = {"panel_md5": digest(panel_path, "md5"),
            "sha256": {f: digest(ROOT / f) for f in ["scripts/44_stage2_data_audit.py", "src/ppmi_subtype/stage1_checks.py",
                                                    "src/ppmi_subtype/raw_scores.py",
                                                    "src/ppmi_subtype/subtype.py"]}}
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    if conf["panel_md5"] != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    res = {}
    panel = pd.read_csv(panel_path, low_memory=False)
    part2, part3 = loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg)
    table = RS.item_visit_table(part2, part3, s)
    frame = RS.build_visit_frame(panel, table, s)
    valid = RS.validate_reconstruction(frame)
    assert valid["exact"], valid
    res["reconstruction"] = valid
    items_t, items_p = list(s.tremor_items), list(s.pigd_items)
    clean = RS.KEY + items_t + items_p
    it = table[RS.KEY].join(pd.DataFrame(
        {c: pd.to_numeric(table[c], errors="coerce") for c in items_t + items_p}))
    lab = frame[frame.subtype_name.notna()].merge(it[clean], on=RS.KEY, how="left")
    lab = lab.sort_values(["PATNO", "years_from_baseline"]).reset_index(drop=True)
    # items of labelled visits are complete, valid integers, and sum to the sums
    blk = lab[items_t + items_p]
    res["labelled_items"] = {"visits": len(lab), "any_missing_item": int(blk.isna().any(axis=1).sum()),
                             "outside_0_4": int(((blk < 0) | (blk > 4)).sum().sum()),
                             "non_integer": int((blk % 1 != 0).sum().sum())}
    assert np.array_equal(lab[items_t].sum(1).to_numpy(), lab.tremor_sum.to_numpy())
    assert np.array_equal(lab[items_p].sum(1).to_numpy(), lab.pigd_sum.to_numpy())
    lab["L"] = lab.subtype_name.map(LAB)
    n_lab = lab.groupby("PATNO").size()
    lab["primary"] = lab.PATNO.map(n_lab).ge(2)
    prim = lab[lab.primary].reset_index(drop=True)
    data = hbayes.build_growth_data(panel)
    res["cohorts"] = {
        "labelled_visits": len(lab), "labelled_patients": int(lab.PATNO.nunique()),
        "primary_patients_ge2_labelled": int(prim.PATNO.nunique()), "primary_visits": len(prim),
        "d1_ratio_defined_patients": data.n_patients, "d1_ratio_defined_visits": data.n_obs,
        "primary_pigd0_visits": int((prim.pigd_sum == 0).sum()),
        "primary_pigd0_TD": int(((prim.pigd_sum == 0) & (prim.tremor_sum > 0)).sum()),
        "primary_both_zero": int(((prim.pigd_sum == 0) & (prim.tremor_sum == 0)).sum()),
        "primary_tremor0_pigd_pos": int(((prim.tremor_sum == 0) & (prim.pigd_sum > 0)).sum()),
        "primary_label_counts": prim.subtype_name.value_counts().to_dict(),
        "unlabelled_reasons": frame.loc[frame.subtype_name.isna(), "reason"].value_counts().to_dict()}

    # ---- item-level distributions ------------------------------------------------------
    rows = []
    for c in items_t + items_p:
        v = prim[c].astype(int)
        cnt = np.bincount(v, minlength=5)
        rows.append({"item": c, "subscore": "tremor" if c in items_t else "pigd",
                     "part": "II" if c.startswith("NP2") else "III",
                     **{f"cat{k}": int(cnt[k]) for k in range(5)},
                     "positive_rate": float((v > 0).mean()), "mean": float(v.mean())})
    items_df = pd.DataFrame(rows)
    items_df.to_csv(out / "A1_item_categories.csv", index=False)
    res["items"] = rows
    # inter-item association (Spearman) within each subscore
    for name, cols in (("tremor", items_t), ("pigd", items_p)):
        cm = prim[cols].corr(method="spearman")
        cm.to_csv(out / f"A2_item_spearman_{name}.csv")
        tri = cm.to_numpy()[np.triu_indices(len(cols), 1)]
        res[f"spearman_{name}"] = {"min": float(tri.min()), "median": float(np.median(tri)),
                                   "max": float(tri.max())}

    # ---- sums, zeros, joint lattice ----------------------------------------------------
    T_, P_ = prim.tremor_sum.astype(int), prim.pigd_sum.astype(int)
    res["sum_ranges"] = {"tremor_observed": [int(T_.min()), int(T_.max())], "tremor_possible": [0, 44],
                         "pigd_observed": [int(P_.min()), int(P_.max())], "pigd_possible": [0, 20],
                         "tremor_zero_share": float((T_ == 0).mean()),
                         "pigd_zero_share": float((P_ == 0).mean()),
                         "tremor_quantiles": np.quantile(T_, [.1, .25, .5, .75, .9, .99]).tolist(),
                         "pigd_quantiles": np.quantile(P_, [.1, .25, .5, .75, .9, .99]).tolist(),
                         "spearman_T_P": float(sstats.spearmanr(T_, P_)[0])}
    pd.crosstab(T_, P_).to_csv(out / "A3_joint_tremor_pigd_counts.csv")
    byP = prim.groupby("pigd_sum").subtype_name.value_counts().unstack(fill_value=0)
    byP["n"] = byP.sum(1)
    byP.to_csv(out / "A4_labels_by_pigd_sum.csv")
    res["labels_by_pigd_sum_head"] = byP.head(8).reset_index().to_dict("records")
    # dispersion of each sum given its patient mean (overdispersion vs binomial on 4*K)
    disp = {}
    for name, x, K in (("tremor", T_, 11), ("pigd", P_, 5)):
        g = x.groupby(prim.PATNO)
        m = g.transform("mean")
        nn = g.transform("size")
        sel = nn >= 3
        resid_var = float(((x - m)[sel] ** 2).sum() / (sel.sum() - g.ngroups))
        pbar = (m[sel] / (4 * K)).clip(1e-9, 1 - 1e-9)
        binom = float((4 * K * pbar * (1 - pbar)).mean())
        disp[name] = {"within_patient_var": resid_var, "binomial_var_at_patient_mean": binom,
                      "ratio": resid_var / binom}
    res["within_patient_dispersion_ge3_visits"] = disp

    # ---- medication / exam state --------------------------------------------------------
    st = prim.PDSTATE.astype("string").str.upper().fillna("NONE")
    tr = pd.to_numeric(prim.get(s.treatment_column), errors="coerce")
    res["medication"] = {"PDSTATE": st.value_counts().to_dict(),
                         "PDTRTMNT": tr.value_counts(dropna=False).to_dict(),
                         "untreated_share_by_year": prim.assign(u=(tr == 0)).groupby(
                             prim.years_from_baseline.round().clip(upper=10)).u.mean().round(3).to_dict()}

    # ---- longitudinal: consecutive labelled visits within patient ---------------------------
    p = prim
    same = (p.PATNO.values[1:] == p.PATNO.values[:-1])
    j = np.flatnonzero(same)
    a, b = p.L.values[j], p.L.values[j + 1]
    dT = p.tremor_sum.values[j + 1] - p.tremor_sum.values[j]
    dP = p.pigd_sum.values[j + 1] - p.pigd_sum.values[j]
    dt = p.years_from_baseline.values[j + 1] - p.years_from_baseline.values[j]
    tr_a, tr_b = tr.values[j], tr.values[j + 1]
    res["pairs"] = {"n": int(len(j)), "interval_median": float(np.median(dt)),
                    "interval_q10_q90": np.quantile(dt, [.1, .9]).tolist(),
                    "dT_skew": float(sstats.skew(dT)), "dP_skew": float(sstats.skew(dP)),
                    "dT_quantiles": np.quantile(dT, [.05, .25, .5, .75, .95]).tolist(),
                    "dP_quantiles": np.quantile(dP, [.05, .25, .5, .75, .95]).tolist(),
                    "corr_dT_dP": float(np.corrcoef(dT, dP)[0, 1]),
                    "dP_given_P0": pd.Series(dP[p.pigd_sum.values[j] == 0]).value_counts().sort_index().to_dict()}
    # direct TD<->PIGD changes on the full label and their outcomes (the tested
    # `stage1_checks.direct_change_outcomes`, whose S3 equals label_statistics);
    # drivers of each change
    pidc = pd.factorize(p.PATNO)[0]
    jv, dv, ov = S1C.direct_change_outcomes(p.L.values, pidc)
    k_of = {int(x): i for i, x in enumerate(j)}         # visit position -> pair index
    Pv, Tv, Lv = p.pigd_sum.values.astype(int), p.tremor_sum.values.astype(int), p.L.values
    p0_patient = p.groupby("PATNO").pigd_sum.apply(lambda x: bool((x == 0).any())).to_dict()
    rows = []
    for jj, d, o in zip(jv, dv, ov):
        k = k_of[int(jj)]
        has_c = o != 3
        rows.append({"direction": S1C.DIRECTIONS[d], "outcome": S1C.OUTCOMES[o],
                     "patient_has_pigd0": bool(p0_patient[p.PATNO.values[jj]]),
                     "dT": int(dT[k]), "dP": int(dP[k]), "dt": float(dt[k]),
                     "P_before": int(Pv[jj]), "P_after": int(Pv[jj + 1]),
                     "P0_involved": bool(Pv[jj] == 0 or Pv[jj + 1] == 0 or (has_c and Pv[jj + 2] == 0)),
                     "treatment_start_in_pair": bool(tr_a[k] == 0 and tr_b[k] == 1),
                     "dT_next": int(Tv[jj + 2] - Tv[jj + 1]) if has_c else None,
                     "dP_next": int(Pv[jj + 2] - Pv[jj + 1]) if has_c else None})
    # cross-check: overall direct reversion on the full label equals label_statistics S3
    from ppmi_subtype import ppc_diagnostic as PD
    s3 = PD.label_statistics(Lv[None, :].astype(np.int8), pidc)[0][2]
    assert np.isclose((ov == 0).sum() / (ov < 3).sum(), s3)
    res["full_label_S3"] = float(s3)
    dc = pd.DataFrame(rows)
    dc.to_csv(out / "A5_direct_changes_full_label.csv", index=False)
    ob = dc[dc.outcome != "unobserved"]
    tab = pd.crosstab(dc.direction, dc.outcome)
    res["direct_full_label"] = {
        "crosstab": tab.to_dict(),
        "R_D_by_direction": (ob.groupby("direction").outcome.apply(lambda x: (x == "revert").mean())).to_dict(),
        "n_with_outcome": ob.groupby("direction").size().to_dict(),
        "driver_share": dc.assign(driver=np.select([(dc.dP != 0) & (dc.dT == 0), (dc.dT != 0) & (dc.dP == 0)],
                                                   ["P only", "T only"], "both"))
                          .groupby("direction").driver.value_counts().unstack(fill_value=0).to_dict(),
        "median_dT_dP": dc.groupby("direction")[["dT", "dP"]].median().to_dict(),
        "revert_rate_by_P0_involved": ob.groupby(["direction", "P0_involved"]).outcome.apply(
            lambda x: f"{(x == 'revert').sum()}/{len(x)}").to_dict(),
        "revert_rate_by_patient_has_pigd0": ob.groupby(["direction", "patient_has_pigd0"]).outcome.apply(
            lambda x: f"{(x == 'revert').sum()}/{len(x)}").to_dict(),
        "patients_with_pigd0_visit": int(sum(p0_patient.values())),
        "TD_to_PIGD_by_dP_sign": ob[ob.direction == "TD->PIGD"].assign(
            up=np.sign(ob[ob.direction == "TD->PIGD"].dP)).groupby("up").outcome.apply(
            lambda x: f"{(x == 'revert').sum()}/{len(x)}").to_dict(),
        "treatment_start_in_pair": dc.treatment_start_in_pair.groupby(dc.direction).sum().to_dict()}
    # medication-state stability across pairs
    res["pairs"]["treatment_start_pairs"] = int(((tr_a == 0) & (tr_b == 1)).sum())
    (out / "results.json").write_text(json.dumps(_keys(res), indent=2, default=_json))
    print(json.dumps({k: res[k] for k in ("cohorts", "labelled_items", "sum_ranges")}, indent=1, default=_json))
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"done in {time.time() - t0:.1f}s")
    sys.exit(rc)
