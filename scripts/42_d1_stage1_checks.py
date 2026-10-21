"""Stage 1 descriptive checks after D1 (audit doc 13, section 5; report doc 14).

Descriptive only: no model is fitted, no cohort is simulated, nothing frozen is
read for writing. Inputs:

* the frozen panel (MD5 checked) and the Part C item reconstruction
  (`raw_scores.build_visit_frame`, validated exactly against the panel);
* the frozen D1 generator draws `fullcohort_tt_draws.npz` (SHA-256 checked
  against `00_generator.json`) and the first 100 saved null-cohort result files.

Model-based references are expectations under the D1 generator for the SAME
posterior draws that generated null cohorts 0-99 (draw index recovered
from each cohort's frozen entropy, no data generated), by quadrature
(`stage1_checks`). They are not the patient-conditional replicates D1 compared
the observed cohort with; those were never saved (see the report).

    python scripts/42_d1_stage1_checks.py --out experiments/<UTC>_d1_stage1_checks
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import loaders  # noqa: E402
from ppmi_subtype import ppc_diagnostic as D  # noqa: E402
from ppmi_subtype import raw_scores as RS  # noqa: E402
from ppmi_subtype import stage1_checks as S  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED, B = 42, 2000
N_THETA = 100        # generator draws of null cohorts 0..99 (quadrature cost ~2 s per draw)
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
D1 = ROOT / "experiments/20261016T070746Z_d1_full/results"
CODE = ["src/ppmi_subtype/stage1_checks.py", "scripts/42_d1_stage1_checks.py",
        "src/ppmi_subtype/raw_scores.py", "src/ppmi_subtype/ppc_diagnostic.py"]
P_STRATA = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 99))
CDF_CUTS = (0.5, 0.90, 1.15, 1.5, 2.0, 3.0)
MED_STRATA = (("1-2", 0, 2.5), ("3-4", 2.5, 4.5), ("5-7", 4.5, 7.5), ("8+", 7.5, 99))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def code_of(values, strata):
    out = np.full(len(values), -1, int)
    for k, (_, lo, hi) in enumerate(strata):
        out[(values >= lo) & (values <= hi)] = k
    return out


def boot(pid, cols, M):
    return RS.patient_bootstrap_sums(S.per_patient(pid, cols, M), B, SEED)


def share_row(name, num, den, bs, i, j):
    lo, hi = S.bootstrap_ratio(bs, i, j)
    return {"stratum": name, "count": int(num), "n": int(den),
            "share": num / den if den else np.nan, "ci_lo": lo, "ci_hi": hi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()
    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    gen = D1 / "fullcohort_tt_draws.npz"
    config = {"sha256": {f: digest(ROOT / f) for f in CODE},
              "panel_md5": digest(panel_path, "md5"), "generator_sha256": digest(gen),
              "seed": SEED, "bootstrap": B, "quadrature": {"n_b0": 40, "n_b1": 32}, "n_theta": N_THETA,
              "d1_run": str(D1.relative_to(ROOT))}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    if config["panel_md5"] != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    if config["generator_sha256"] != json.loads((D1 / "00_generator.json").read_text())["npz_sha256"]:
        print("generator draws differ from the frozen file", file=sys.stderr)
        return 2

    # ---- cohort: exactly D1's ratio-defined visits, with item sums --------------------
    panel = pd.read_csv(panel_path, low_memory=False)
    part2, part3 = loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg)
    frame = RS.build_visit_frame(panel, RS.item_visit_table(part2, part3, cfg.subtype), cfg.subtype)
    valid = RS.validate_reconstruction(frame)
    if not valid["exact"]:
        print(f"reconstruction mismatch {valid}", file=sys.stderr)
        return 3
    data = hbayes.build_growth_data(panel)
    v = frame[frame.ratio.notna() & np.isfinite(frame.ratio) & frame.PATNO.isin(data.patients)]
    v = v.sort_values(["PATNO", "years_from_baseline"]).reset_index(drop=True)
    assert len(v) == data.n_obs and np.allclose(v.ratio, data.y) and np.allclose(v.years_from_baseline, data.t)
    pid, t, y = data.patient_index, data.t, data.y
    T_, P_ = v.tremor_sum.to_numpy(int), v.pigd_sum.to_numpy(int)
    assert np.allclose((T_ / 11) / (P_ / 5), y), "ratio is not 5T/(11P) on every visit"
    M, N = data.n_patients, data.n_obs
    L = D.labels_from_ratio(y)
    obs_d1 = json.loads((D1 / "05_observed.json").read_text())["observed"]["S_obs"]
    S_here = D.label_statistics(L[None, :], pid)[0]
    assert np.allclose(S_here[:5], [obs_d1[k] for k in D.STAT_NAMES])
    log(f"cohort: {M} patients, {N} visits; statistics equal 05_observed.json")

    # labelled visits excluded for PIGD sum 0, within D1 patients
    lab = frame[frame.subtype_name.notna() & frame.PATNO.isin(data.patients)]
    excl = lab[lab.pigd_score == 0]
    affected_p = np.isin(data.patients, excl.PATNO.unique())
    affected = affected_p[pid].astype(int)
    # gap-spanning adjacency: an excluded visit lies between visits j and j+1
    s1, s2 = S.neighbours(pid)
    gap = np.zeros(N, bool)
    ex_t = excl.groupby("PATNO").years_from_baseline.apply(np.sort).to_dict()
    for j in np.flatnonzero(s1):
        ts = ex_t.get(data.patients[pid[j]])
        if ts is not None and np.any((ts > t[j]) & (ts < t[j + 1])):
            gap[j] = True
    gap_trip = gap | np.r_[gap[1:], False] & s2     # either step of the triplet spans one
    pstrat = code_of(P_, P_STRATA)
    dest_strat = np.r_[pstrat[1:], -1]                # stratum of the visit after j

    # ---- model-based: generator expectations at the null-cohort draws ---------------
    z = np.load(gen)
    post = type("P", (), {k: z[k] for k in z.files})
    idx = np.array([int(np.random.default_rng([42, 41, 0, c]).integers(len(z["sigma2"])))
                    for c in range(N_THETA)])
    null_S = [json.loads((D1 / f"cohorts/null/null_{c:03d}.json").read_text())["S_obs"]
              for c in range(N_THETA)]
    grid = RS.label_grid(cfg.subtype)
    lo_lat, hi_lat = S.catchment_bounds(grid, P_)
    parts = {"all": np.zeros(N, int), "pstrat": pstrat, "dest": np.where(dest_strat < 0, 0, dest_strat),
             "affected": affected, "gap": gap.astype(int), "gap_trip": gap_trip.astype(int)}
    sizes = {k: int(c.max()) + 1 for k, c in parts.items()}
    qnames = (["ind"] + [f"pair{d}" for d in (0, 1)] + [f"unobs{d}" for d in (0, 1)]
              + [f"{o}{d}" for d in (0, 1) for o in ("revert", "stay", "to_Ind")])
    tot = {var: {k: np.zeros((N_THETA, len(qnames), sizes[k])) for k in parts} for var in ("band", "lattice")}
    ivar = {var: {k: np.zeros((N_THETA, sizes[k])) for k in ("pstrat", "affected", "all")}
            for var in ("band", "lattice")}
    cdf = np.zeros((N_THETA, len(CDF_CUTS)))
    t0 = time.time()
    for r, di in enumerate(idx):
        th = D.theta_from(post, int(di))
        cdf[r] = S.expected_cdf(t, th, CDF_CUTS)
        for var, (lo, hi) in (("band", (None, None)), ("lattice", (lo_lat, hi_lat))):
            ex = S.expected_visit_quantities(t, pid, th, lo, hi,
                                             groups={gk: parts[gk] for gk in ivar[var]})
            for gk in ivar[var]:
                ivar[var][gk][r] = ex["ind_var"][gk]
            q = np.vstack([ex["ind"], ex["pair"][0], ex["pair"][1], ex["unobs"][0], ex["unobs"][1]]
                          + [ex["trip"][d][o] for d in (0, 1) for o in ("revert", "stay", "to_Ind")])
            for k, c in parts.items():
                tot[var][k][r] = np.stack([np.bincount(c, weights=row, minlength=sizes[k]) for row in q])
        if r % 20 == 0:
            log(f"quadrature {r}/{N_THETA} ({time.time() - t0:.0f}s)")
    qi = {n: i for i, n in enumerate(qnames)}

    def model(var, part, qn, g=0):
        x = tot[var][part][:, qi[qn], g]
        return float(x.mean()), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))

    def model_ratio(var, part, num, den, g=0):
        n_ = sum(tot[var][part][:, qi[q], g] for q in num)
        d_ = sum(tot[var][part][:, qi[q], g] for q in den)
        r_ = n_ / d_
        return float(n_.mean() / d_.mean()), float(np.quantile(r_, 0.025)), float(np.quantile(r_, 0.975))

    def ind_pred(var, part, g):
        e = tot[var][part][:, qi["ind"], g]
        sd = np.sqrt(ivar[var][part][:, g].mean() + e.var())
        return float(e.mean()), float(sd)

    res = {"cohort": {"patients": M, "visits": N, "affected_patients": int(affected_p.sum()),
                      "excluded_pigd0_labelled_visits": int(len(excl)),
                      "excluded_by_label": excl.subtype_name.value_counts().to_dict(),
                      "gap_spanning_adjacent_pairs": int(gap.sum()),
                      "reconstruction": valid}}

    # ---- validation of the quadrature against the saved null cohorts ---------------------
    S1_null = np.array([s["S1_ind_share"] for s in null_S])
    S3_null = np.array([s["S3_R_D"] for s in null_S])
    e_S1 = tot["band"]["all"][:, qi["ind"], 0] / N
    rev = sum(tot["band"]["all"][:, qi[f"revert{d}"], 0] for d in (0, 1))
    den = sum(tot["band"]["all"][:, qi[f"{o}{d}"], 0] for d in (0, 1) for o in ("revert", "stay", "to_Ind"))
    sd_S1 = np.sqrt(ivar["band"]["all"][:, 0]) / N
    res["validation"] = {
        "null_S1_realised_mean": float(S1_null.mean()), "null_S1_expected_mean": float(e_S1.mean()),
        "corr_expected_vs_realised_S1": float(np.corrcoef(e_S1, S1_null)[0, 1]),
        "S1_standardised_resid_mean": float(((S1_null - e_S1) / sd_S1).mean()),
        "S1_standardised_resid_sd": float(((S1_null - e_S1) / sd_S1).std()),
        "null_S3_realised_mean": float(np.nanmean(S3_null)),
        "null_S3_ratio_of_expectations": float(rev.mean() / den.mean()),
        "corr_expected_vs_realised_S3": float(np.corrcoef(rev / den, S3_null)[0, 1])}
    log(f"validation {res['validation']}")

    # ---- Check 1: Indeterminate share by PIGD sum and lattice cell ------------------------
    ind = L == S.IND
    cols = np.column_stack([ind & (pstrat == k) for k in range(len(P_STRATA))]
                           + [pstrat == k for k in range(len(P_STRATA))] + [ind, np.ones(N)])
    bs = boot(pid, cols, M)
    K = len(P_STRATA)
    rows = []
    total_excess = {var: ind.sum() - model(var, "all", "ind")[0] for var in ("band", "lattice")}
    for k, (name, _, _) in enumerate(P_STRATA + (("all", 0, 0),)):
        g = 0 if name == "all" else k
        part = "all" if name == "all" else "pstrat"
        n_k = N if name == "all" else int((pstrat == k).sum())
        o_k = int(ind.sum() if name == "all" else (ind & (pstrat == k)).sum())
        row = share_row(name, o_k, n_k, bs, 2 * K if name == "all" else k, 2 * K + 1 if name == "all" else K + k)
        for var in ("band", "lattice"):
            e, sd = ind_pred(var, part, g)
            row.update({f"{var}_expected": e, f"{var}_pred_lo": e - 1.96 * sd,
                        f"{var}_pred_hi": e + 1.96 * sd, f"{var}_excess": o_k - e,
                        f"{var}_excess_share_of_total": (o_k - e) / total_excess[var]})
        if name != "all":
            row["inflation_at_lowest_sum"] = S.lattice_row(grid, P_STRATA[k][1])["inflation"]
        rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(out / "check1_ind_by_pigd_stratum.csv", index=False)
    cells = (v.assign(label=L).groupby(["pigd_sum", "tremor_sum"]).label
             .agg(n="size", ind=lambda s: int((s == 1).sum()), td=lambda s: int((s == 0).sum()),
                  pigd=lambda s: int((s == 2).sum())).reset_index())
    cells["ratio"] = (cells.tremor_sum / 11) / (cells.pigd_sum / 5)
    cells.to_csv(out / "check1_cells.csv", index=False)
    ind_cells = cells[cells.ind > 0].sort_values("ind", ascending=False)
    lat = pd.DataFrame([S.lattice_row(grid, p) for p in range(1, 21)])
    lat.to_csv(out / "check1_lattice_rows.csv", index=False)
    cdf_rows = [{"cut": c, "observed_share": float((y <= c).mean()),
                 "generator_share": float(cdf[:, k].mean() / N),
                 "generator_lo": float(np.quantile(cdf[:, k], 0.025) / N),
                 "generator_hi": float(np.quantile(cdf[:, k], 0.975) / N)}
                for k, c in enumerate(CDF_CUTS)]
    pd.DataFrame(cdf_rows).to_csv(out / "check1_ratio_cdf.csv", index=False)
    dl = pd.DataFrame(S.delatticed_ind(P_, T_, grid))
    dl.to_csv(out / "check1_delatticed.csv", index=False)
    res["check1_delatticed"] = {"rows": dl.to_dict("records"), "ind_total": int(dl.ind.sum()),
                                "delatticed_total": float(dl.delatticed.sum()),
                                "delatticed_share": float(dl.delatticed.sum() / N)}
    res["check1"] = {"table": t1.to_dict("records"), "ratio_cdf": cdf_rows,
                     "top_ind_cells": ind_cells.head(15).to_dict("records"),
                     "ind_at_0909_or_1136": int(cells.loc[np.isclose(cells.ratio, 10 / 11)
                                                          | np.isclose(cells.ratio, 25 / 22), "ind"].sum()),
                     "n_distinct_ind_ratios": int(cells.loc[cells.ind > 0, "ratio"].round(9).nunique()),
                     "lattice_rows": lat.head(8).to_dict("records")}

    # ---- Check 2: direct changes by direction and outcome ---------------------------------
    j, dirn, outc = S.direct_change_outcomes(L, pid)
    assert np.isclose((outc == 0).sum() / (outc < 3).sum(), S_here[2])
    ev = pd.DataFrame({"j": j, "direction": np.array(S.DIRECTIONS)[dirn],
                       "outcome": np.array(S.OUTCOMES)[outc], "dest_stratum": dest_strat[j],
                       "affected": affected[j], "gap_pair": gap[j], "gap_trip": gap_trip[j]})
    ev.to_csv(out / "check2_direct_changes.csv", index=False)
    tab2 = pd.crosstab(ev.direction, ev.outcome).reindex(columns=list(S.OUTCOMES), fill_value=0)
    rows2 = []
    for d, dname in enumerate(S.DIRECTIONS):
        oc = [(dirn == d) & (outc == o) for o in range(4)]
        cols = np.zeros((N, 4))
        for o in range(4):
            cols[j[oc[o]], o] = 1
        cols2 = np.column_stack([cols[:, 0], cols[:, :3].sum(1)])
        bsd = boot(pid, cols2, M)
        n_obs_out = int(cols[:, :3].sum())
        row = {"direction": dname, **{S.OUTCOMES[o]: int(oc[o].sum()) for o in range(4)},
               "R_D": cols[:, 0].sum() / n_obs_out if n_obs_out else np.nan}
        row["R_D_ci"] = S.bootstrap_ratio(bsd, 0, 1)
        for var in ("band", "lattice"):
            for o in ("revert", "stay", "to_Ind", "unobs"):
                row[f"{var}_E_{o}"] = model(var, "all", f"{o}{d}")[0]
            row[f"{var}_R_D"] = model_ratio(var, "all", [f"revert{d}"],
                                            [f"{o}{d}" for o in ("revert", "stay", "to_Ind")])
        rows2.append(row)
    both = {"direction": "both", **{o: int((outc == k).sum()) for k, o in enumerate(S.OUTCOMES)},
            "R_D": float(S_here[2])}
    for var in ("band", "lattice"):
        for o in ("revert", "stay", "to_Ind", "unobs"):
            both[f"{var}_E_{o}"] = sum(model(var, "all", f"{o}{d}")[0] for d in (0, 1))
        both[f"{var}_R_D"] = model_ratio(var, "all", ["revert0", "revert1"],
                                         [f"{o}{d}" for d in (0, 1) for o in ("revert", "stay", "to_Ind")])
    rows2.append(both)
    by_dest = (ev[ev.outcome != "unobserved"].groupby(["direction", "dest_stratum"]).outcome
               .agg(n="size", revert=lambda s: int((s == "revert").sum())).reset_index())
    by_dest["dest_stratum"] = by_dest.dest_stratum.map(dict(enumerate(n for n, _, _ in P_STRATA)))
    res["check2"] = {"table": rows2, "crosstab": tab2.to_dict(), "by_destination_pigd_stratum":
                     by_dest.to_dict("records")}
    pd.DataFrame(rows2).to_csv(out / "check2_direction_outcome.csv", index=False)

    # ---- Check 3: residual-shape proxy (no fit; population beta1 only) --------------------
    beta1 = float(np.median(z["beta"][:, 1]))
    r = S.centred_detrended(y, t, pid, beta1)
    n_i = np.bincount(pid)
    med_p = pd.Series(P_).groupby(pid).median().to_numpy()
    mstrat = code_of(med_p, MED_STRATA)[pid]
    keep = n_i[pid] >= 3                     # 2-visit patients give mirror pairs: symmetric by construction
    rng = np.random.default_rng(SEED)
    measures = {"bowley": S.bowley, "tail_asym_05": lambda x: S.tail_asymmetry(x, 0.05),
                "tail_asym_01": lambda x: S.tail_asymmetry(x, 0.01),
                "frac_pos": lambda x: float((x > 0).mean())}
    rows3 = []
    for k, (name, _, _) in enumerate(MED_STRATA + (("all", 0, 0),)):
        sel = keep & ((mstrat == k) if name != "all" else True)
        pats = np.unique(pid[sel])
        by_p = {p: np.flatnonzero(sel & (pid == p)) for p in pats}
        row = {"stratum": name, "patients": int(len(pats)), "visits": int(sel.sum()),
               "median_pigd_sum": float(np.median(P_[sel])),
               "step_at_median": float(5 / (11 * np.median(P_[sel]))),
               "iqr": float(np.subtract(*np.quantile(r[sel], [0.75, 0.25])))}
        bt = {m: [] for m in measures}
        for _ in range(1000):                # patient bootstrap
            xs = r[np.concatenate([by_p[p] for p in rng.choice(pats, len(pats))])]
            for m, f in measures.items():
                bt[m].append(f(xs))
        for m, f in measures.items():
            row[m] = f(r[sel])
            row[f"{m}_ci"] = (float(np.nanquantile(bt[m], 0.025)), float(np.nanquantile(bt[m], 0.975)))
        rows3.append(row)
    res["check3"] = {"beta1_median": beta1, "table": rows3, "model_implied": {"bowley": 0.0,
                     "tail_asym_05": 0.0, "tail_asym_01": 0.0, "frac_pos": 0.5},
                     "n_patients_with_2_visits_excluded": int((n_i == 2).sum())}
    pd.DataFrame(rows3).to_csv(out / "check3_residual_proxy.csv", index=False)

    # ---- Check 4: affected vs unaffected patients --------------------------------------
    rows4 = []
    for g, name in ((1, "affected"), (0, "unaffected")):
        selv = affected == g
        cols = np.column_stack([ind & selv, selv])
        ev_g = ev[(ev.affected == g) & (ev.outcome != "unobserved")]
        cnt = np.zeros((N, 2))
        cnt[ev_g.j[ev_g.outcome == "revert"], 0] = 1
        cnt[ev_g.j, 1] = 1
        bs4 = boot(pid, np.column_stack([cols, cnt]), M)
        row = {"group": name, "patients": int(len(np.unique(pid[selv]))), "visits": int(selv.sum()),
               "ind": int((ind & selv).sum()), "S1": (ind & selv).sum() / selv.sum(),
               "S1_ci": S.bootstrap_ratio(bs4, 0, 1), "direct_with_outcome": int(len(ev_g)),
               "reverted": int((ev_g.outcome == "revert").sum()),
               "R_D": (ev_g.outcome == "revert").mean() if len(ev_g) else np.nan,
               "R_D_ci": S.bootstrap_ratio(bs4, 2, 3)}
        for var in ("band", "lattice"):
            e, sd = ind_pred(var, "affected", g)
            row[f"{var}_E_ind"], row[f"{var}_ind_pred_sd"] = e, sd
            row[f"{var}_E_S1"] = e / selv.sum()
            row[f"{var}_R_D"] = model_ratio(var, "affected", ["revert0", "revert1"],
                                            [f"{o}{d}" for d in (0, 1) for o in ("revert", "stay", "to_Ind")], g)
        rows4.append(row)
    rows4b = []
    for g, name in ((1, "triplet spans an excluded visit"), (0, "no excluded visit in triplet")):
        ev_g = ev[(ev.gap_trip == bool(g)) & (ev.outcome != "unobserved")]
        row = {"transition_group": name, "direct_with_outcome": int(len(ev_g)),
               "reverted": int((ev_g.outcome == "revert").sum()),
               "R_D": (ev_g.outcome == "revert").mean() if len(ev_g) else np.nan}
        for var in ("band", "lattice"):
            row[f"{var}_R_D"] = model_ratio(var, "gap_trip", ["revert0", "revert1"],
                                            [f"{o}{d}" for d in (0, 1) for o in ("revert", "stay", "to_Ind")], g)
            row[f"{var}_E_direct_with_outcome"] = sum(
                model(var, "gap_trip", f"{o}{d}", g)[0] for d in (0, 1) for o in ("revert", "stay", "to_Ind"))
        rows4b.append(row)
    res["check4"] = {"patients": rows4, "transitions": rows4b}
    pd.DataFrame(rows4).to_csv(out / "check4_affected.csv", index=False)
    pd.DataFrame(rows4b).to_csv(out / "check4_transitions.csv", index=False)

    figures(out, t1, rows2, rows3, r, mstrat, keep, rows4, cdf_rows)
    res["runtime_s"] = time.time() - t_start
    (out / "results.json").write_text(json.dumps(res, indent=2, default=_json))
    log(f"done in {res['runtime_s']:.0f}s -> {out}")
    return 0


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def figures(out, t1, rows2, rows3, r, mstrat, keep, rows4, cdf_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir()
    # F1: Indeterminate share by PIGD-sum stratum
    f = t1[t1.stratum != "all"]
    x = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x - 0.15, f.share, yerr=[f.share - f.ci_lo, f.ci_hi - f.share], fmt="o", c="k",
                capsize=3, label="observed (patient-bootstrap 95% CI)")
    for off, var, c in ((0.0, "band", "C0"), (0.15, "lattice", "C1")):
        e = f[f"{var}_expected"] / f.n
        ax.errorbar(x + off, e, yerr=[e - f[f"{var}_pred_lo"] / f.n, f[f"{var}_pred_hi"] / f.n - e],
                    fmt="s", c=c, capsize=3,
                    label={"band": "generator, continuous band", "lattice": "generator, lattice catchment [H]"}[var])
    ax.set_xticks(x, [f"{s}\n(n={n})" for s, n in zip(f.stratum, f.n)])
    ax.set_xlabel("PIGD item sum at the visit")
    ax.set_ylabel("Indeterminate share")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "F1_ind_by_pigd_sum.png", dpi=150)
    plt.close(fig)
    # F1b: ratio mass per interval, observed vs generator
    cut = [0.0] + [c["cut"] for c in cdf_rows]
    o = np.diff([0.0] + [c["observed_share"] for c in cdf_rows])
    g = np.diff([0.0] + [c["generator_share"] for c in cdf_rows])
    w = np.diff(cut)
    lab = [f"({a:g}, {b:g}]" if a > 0 else f"<= {b:g}" for a, b in zip(cut[:-1], cut[1:])]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xx = np.arange(len(w))
    ax.bar(xx - 0.2, o / w, 0.4, color="k", label="observed")
    ax.bar(xx + 0.2, g / w, 0.4, color="C0", label="generator (mean of 100 draws)")
    ax.set_xticks(xx, lab, fontsize=8)
    ax.set_xlabel("ratio interval (0.90-1.15 is the Indeterminate band)")
    ax.set_ylabel("share of visits per unit ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "F1b_ratio_density_by_interval.png", dpi=150)
    plt.close(fig)
    # F2: direction x outcome
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, row in zip(axes, rows2[:2]):
        oc = ["revert", "stay", "to_Ind"]
        xx = np.arange(3)
        ax.bar(xx - 0.27, [row[o] for o in oc], 0.27, color="k", label="observed")
        ax.bar(xx, [row[f"band_E_{o}"] for o in oc], 0.27, color="C0", label="generator, band")
        ax.bar(xx + 0.27, [row[f"lattice_E_{o}"] for o in oc], 0.27, color="C1", label="generator, lattice [H]")
        ax.set_xticks(xx, oc)
        ax.set_title(f"{row['direction']} (outcome observed)")
    axes[0].set_ylabel("count of direct changes")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "F2_direct_change_direction_outcome.png", dpi=150)
    plt.close(fig)
    # F3: centred residual proxy by patient-median PIGD stratum
    fig, ax = plt.subplots(figsize=(7, 4))
    q = np.linspace(0.01, 0.25, 25)
    for k, (name, _, _) in enumerate(MED_STRATA + (("all", 0, 0),)):
        xk = r[keep & ((mstrat == k) if name != "all" else True)]
        lo_q, hi_q = np.quantile(xk, q), np.quantile(xk, 1 - q)
        ax.plot(q, (hi_q + lo_q) / (hi_q - lo_q), "-o" if name == "all" else "-", ms=3,
                c="k" if name == "all" else f"C{k}",
                label="all patients" if name == "all" else f"median PIGD sum {name}")
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_xlabel("tail probability q")
    ax.set_ylabel("(Q(1-q) + Q(q)) / (Q(1-q) - Q(q))\n0 for the symmetric generator")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "F3_residual_proxy_symmetry.png", dpi=150)
    plt.close(fig)
    # F4: affected vs unaffected
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (stat, obs_ci, keyb, keyl) in zip(axes, (("S1", "S1_ci", "band_E_S1", "lattice_E_S1"),
                                                    ("R_D", "R_D_ci", "band_R_D", "lattice_R_D"))):
        for i, row in enumerate(rows4):
            o = row[stat]
            ax.errorbar(i - 0.15, o, yerr=[[o - row[obs_ci][0]], [row[obs_ci][1] - o]], fmt="o", c="k", capsize=3)
            b = row[keyb][0] if isinstance(row[keyb], tuple) else row[keyb]
            lt = row[keyl][0] if isinstance(row[keyl], tuple) else row[keyl]
            ax.plot(i, b, "s", c="C0")
            ax.plot(i + 0.15, lt, "s", c="C1")
        ax.set_xticks([0, 1], [f"{row['group']}\n({row['patients']} pts)" for row in rows4])
        ax.set_title(stat)
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([], [], c="k", marker="o", ls=""), Line2D([], [], c="C0", marker="s", ls=""),
                    Line2D([], [], c="C1", marker="s", ls="")],
                   ["observed (patient-bootstrap 95% CI)", "generator, band", "generator, lattice [H]"],
                   fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "F4_pigd0_affected.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
