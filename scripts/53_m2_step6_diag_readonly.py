"""EXPLORATORY DIAGNOSTIC (Step 6 failure investigation, Phase A): read-only D1-D3.

Reads only saved, hash-verified outputs of the pilot, Step 4, Step 5 and Step 6 (no model is fitted,
no data generated). Writes to <out>/ only. Proposal: docs/cluster_phase/39_..._PROPOSED_20261019.md.

  D1  affine latent-scale alignment of item estimates (N(0,1) stage 1, S1-b, Step 5 true-latent "oracle")
      against the generating items; the ψ error that the fitted relabelling implies, vs the observed one.
  D2  error decomposition: Step 4 (true items) -> ψ̂(1) (N(0,1) items) -> ψ̂(2) (S1-b items); Step 5
      first-order shifts against the actual shifts.
  D3  bias, spread and interval diagnostics for all 19 components.

None of this re-scores Step 6 (formal verdict FAIL, doc 38) or replaces its intervals.

    python scripts/53_m2_step6_diag_readonly.py --out <diag dir>/readonly
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import beta as beta_dist
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppmi_subtype import m2_grm as M  # noqa: E402

PILOT = ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results/step1_recovery.json"
S4 = ROOT / "experiments/20261018T184713Z_stage2_step4_fixed_item_recovery/step4_run2"
S5 = ROOT / "experiments/20261018T215102Z_stage2_step5_s1a/step5"
S6 = ROOT / "experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6"
EXPECTED = {S6 / "summary.json": None, S4 / "summary.json":
            "00433150c23c90ecc039bff39fe2cf87d1fa13614b597cad0a611964af7afa39",
            S5 / "summary.json": "5ccc690cd6eb5b5d93ceb06e7d9d7678ee9c3471b6fc247b1f1e46d64b429dae",
            PILOT: "d16d5435b85eae99ff84f22b69594b0853ff23833c983a2b583e082b79188603"}
FAILED = ("beta0_P", "Le00_log", "Le11_log")
Z95 = 1.96


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
#  Affine relabelling θ' = (θ − d)/c:  â = c·a,  τ̂ = (τ − d)/c;  ψ' from ψ correspondingly
# ---------------------------------------------------------------------------
def affine_fit(a_hat, tau_hat, a0, tau0):
    """Three estimates of (c, d): thresholds only (OLS of τ̂ on τ), discriminations only
    (c = geometric mean of â/a; d not identified), and joint least squares on log â − log a − log c
    and τ̂ − (τ − d)/c (each residual unweighted)."""
    a_hat, tau_hat, a0, tau0 = (np.asarray(x, float) for x in (a_hat, tau_hat, a0, tau0))
    sl, ic = np.linalg.lstsq(np.column_stack([tau0.ravel(), np.ones(tau0.size)]), tau_hat.ravel(), rcond=None)[0]
    c_tau, d_tau = 1 / sl, -ic / sl
    c_a = float(np.exp(np.mean(np.log(a_hat / a0))))

    def res(p):
        c, d = np.exp(p[0]), p[1]
        return np.concatenate([np.log(a_hat / a0) - np.log(c), (tau_hat - (tau0 - d) / c).ravel()])
    sol = least_squares(res, [np.log(c_tau), d_tau])
    c_j, d_j = float(np.exp(sol.x[0])), float(sol.x[1])
    resid_tau = tau_hat - (tau0 - d_j) / c_j
    resid_la = np.log(a_hat / a0) - np.log(c_j)
    return {"c_tau": c_tau, "d_tau": d_tau, "c_a": c_a, "c_joint": c_j, "d_joint": d_j,
            "tau_rmse_raw": float(np.sqrt(np.mean((tau_hat - tau0) ** 2))),
            "tau_rmse_after": float(np.sqrt(np.mean(resid_tau ** 2))),
            "loga_rmse_raw": float(np.sqrt(np.mean(np.log(a_hat / a0) ** 2))),
            "loga_rmse_after": float(np.sqrt(np.mean(resid_la ** 2))),
            "a_err_mean": float(np.mean(a_hat - a0)), "tau_err_mean": float(np.mean(tau_hat - tau0)),
            "resid_tau_by_item": np.sqrt(np.mean(resid_tau ** 2, axis=1)), "resid_loga_by_item": resid_la}


def relabel_psi(psi, cT, dT, cP, dP):
    """ψ expressed on the relabelled scale θ'_s = (θ_s − d_s)/c_s (exact; see module doc)."""
    beta, D, _, S, _ = M.unpack(psi)
    b = np.asarray(beta, float).copy()
    b[0], b[1], b[2] = (b[0] - dT) / cT, b[1] / cT, b[2] / cT
    b[3], b[4], b[5] = (b[3] - dP) / cP, b[4] / cP, b[5] / cP
    C4 = np.diag(1 / np.array([cT, cT, cP, cP]))
    C2 = np.diag(1 / np.array([cT, cP]))
    return M.to_psi(b, C4 @ D @ C4, C2 @ S @ C2)


def cp_interval(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    prov = {"label": "EXPLORATORY DIAGNOSTIC — not a recovery test; Step 6 remains FAILED",
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in EXPECTED} |
            {str(p.relative_to(ROOT)): sha(p) for r in range(5) for p in
             (S6 / f"S1_dataset{r}.json", S4 / f"S1_dataset{r}.json", S5 / f"S1_items_dataset{r}.json")} |
            {str((S5 / "S4_downstream_shift.csv").relative_to(ROOT)): sha(S5 / "S4_downstream_shift.csv")},
            "script_sha256": sha(Path(__file__))}
    bad = [str(p) for p, h in EXPECTED.items() if h is not None and sha(p) != h]
    m6 = (S6 / "MANIFEST.sha256").read_text().splitlines()
    bad += [ln.split()[1] for ln in m6 if sha(S6 / ln.split()[1]) != ln.split()[0]]
    prov["integrity_ok"] = not bad
    if bad:
        (out / "STOPPED.json").write_text(json.dumps({"hash mismatch": bad}, indent=1))
        return 2

    s6 = json.loads((S6 / "summary.json").read_text())
    names, truth = s6["names"], np.array(s6["psi_true"])
    J = len(names)
    pilot = json.loads(PILOT.read_text())
    d6 = [json.loads((S6 / f"S1_dataset{r}.json").read_text()) for r in range(5)]
    d4 = [json.loads((S4 / f"S1_dataset{r}.json").read_text()) for r in range(5)]
    d5 = [json.loads((S5 / f"S1_items_dataset{r}.json").read_text()) for r in range(5)]
    true_items = {s: (np.array(d5[0]["true"][s]["a"]), np.array(d5[0]["true"][s]["tau"])) for s in "TP"}
    for r in range(5):                                          # the generating items are the same in every file
        assert all(np.array_equal(d5[r]["true"][s]["a"], true_items[s][0]) for s in "TP")

    est = {"step4": [(np.array(d4[r]["psi_hat"]), np.array(d4[r]["se"])) for r in range(5)],
           "pilot": [(np.array(pilot["datasets"][r]["psi_hat"]), np.array(pilot["datasets"][r]["se"])) for r in range(5)],
           "psi1": [(np.array(d6[r]["stage2_fit_1"]["psi"]), np.array(d6[r]["coverage_1_nonbinding"]["se"]))
                    for r in range(5)],
           "psi2": [(np.array(d6[r]["stage2_fit_2"]["psi"]), np.array(d6[r]["coverage_2"]["se"])) for r in range(5)]}
    items = {"N01": [{s: (d6[r]["items_1"][s]["a"], d6[r]["items_1"][s]["tau"]) for s in "TP"} for r in range(5)],
             "S1b": [{s: (d6[r]["items_2"][s]["a"], d6[r]["items_2"][s]["tau"]) for s in "TP"} for r in range(5)],
             "oracle": [{s: (d5[r]["fits"][f"oracle_{s}"]["a"], d5[r]["fits"][f"oracle_{s}"]["tau"]) for s in "TP"}
                        for r in range(5)]}
    for r in range(5):                                          # Step 5's N01 items are the Step 6 stage-1 items
        for s in "TP":
            assert np.array_equal(d5[r]["fits"][f"N01_{s}"]["a"], d6[r]["items_1"][s]["a"])
    prov["checks"] = {"step5_N01_items_equal_step6_stage1_items": True, "generating_items_identical_across_files": True}

    # ---------------- D1 -------------------------------------------------------------------------
    d1_rows, d1_item_rows, d1_pred_rows = [], [], []
    obs_key = {"N01": "psi1", "S1b": "psi2", "oracle": None}
    for iset in ("N01", "S1b", "oracle"):
        for r in range(5):
            al = {s: affine_fit(*items[iset][r][s], *true_items[s]) for s in "TP"}
            for s in "TP":
                x = al[s]
                d1_rows.append({"item_set": iset, "dataset": r, "subscore": s, **{k: v for k, v in x.items()
                                                                                  if not k.startswith("resid_")}})
                for k in range(len(true_items[s][0])):
                    d1_item_rows.append({"item_set": iset, "dataset": r, "subscore": s, "item": k,
                                         "a_err": items[iset][r][s][0][k] - true_items[s][0][k],
                                         "loga_resid_after_affine": x["resid_loga_by_item"][k],
                                         "tau_rmse_resid_after_affine": x["resid_tau_by_item"][k]})
            for variant, (cT, dT, cP, dP) in {
                    "joint": (al["T"]["c_joint"], al["T"]["d_joint"], al["P"]["c_joint"], al["P"]["d_joint"]),
                    "thresholds": (al["T"]["c_tau"], al["T"]["d_tau"], al["P"]["c_tau"], al["P"]["d_tau"])}.items():
                pred = relabel_psi(truth, cT, dT, cP, dP) - truth
                ok = obs_key[iset]
                psi_o, se_o = est[ok][r] if ok else (None, est["step4"][r][1])
                for j, nm in enumerate(names):
                    d1_pred_rows.append({"item_set": iset, "variant": variant, "dataset": r, "param": nm,
                                         "pred_err": pred[j], "se_ref": se_o[j], "pred_z": pred[j] / se_o[j],
                                         "obs_err": (psi_o[j] - truth[j]) if ok else np.nan,
                                         "obs_z": ((psi_o[j] - truth[j]) / se_o[j]) if ok else np.nan})
    D1 = pd.DataFrame(d1_rows)
    D1i = pd.DataFrame(d1_item_rows)
    D1p = pd.DataFrame(d1_pred_rows)
    D1.to_csv(out / "D1_affine_fits.csv", index=False)
    D1i.to_csv(out / "D1_item_residuals.csv", index=False)
    D1p.to_csv(out / "D1_predicted_vs_observed.csv", index=False)
    agg = []
    for (iset, var, nm), g in D1p[D1p.item_set != "oracle"].groupby(["item_set", "variant", "param"], sort=False):
        o, p = g.obs_z.values, g.pred_z.values
        agg.append({"item_set": iset, "variant": var, "param": nm, "mean_obs_z": o.mean(), "mean_pred_z": p.mean(),
                    "mean_resid_z": (o - p).mean(), "rms_obs_z": np.sqrt(np.mean(o ** 2)),
                    "rms_resid_z": np.sqrt(np.mean((o - p) ** 2)),
                    "explained_share": 1 - np.sum((o - p) ** 2) / np.sum(o ** 2)})
    D1agg = pd.DataFrame(agg)
    D1agg.to_csv(out / "D1_prediction_summary.csv", index=False)

    # ---------------- D2 -------------------------------------------------------------------------
    sh = pd.read_csv(S5 / "S4_downstream_shift.csv")
    d2 = []
    for r in range(5):
        p4, se4 = est["step4"][r]
        p1, se1 = est["psi1"][r]
        p2, se2 = est["psi2"][r]
        pp, sep = est["pilot"][r]
        for j, nm in enumerate(names):
            row5 = sh[(sh.dataset == r) & (sh.param == nm)].iloc[0]
            d2.append({"dataset": r, "param": nm, "truth": truth[j],
                       "z_step4": (p4[j] - truth[j]) / se4[j], "z_pilot": (pp[j] - truth[j]) / sep[j],
                       "z_psi1": (p1[j] - truth[j]) / se1[j], "z_psi2": (p2[j] - truth[j]) / se2[j],
                       "items_effect_z4": (p1[j] - p4[j]) / se4[j],            # ψ̂(1) − ψ̂_Step4, Step 4 SE units
                       "s1b_effect_z2": (p2[j] - p1[j]) / se2[j],              # ψ̂(2) − ψ̂(1), Step 6 SE units
                       "pred_items_effect_z4": row5.contrast_z_N01_minus_true,  # Step 5 first order
                       "pred_assumption_z4": row5.contrast_z_N01_minus_oracle,
                       "pred_sampling_z4": row5.contrast_z_oracle_minus_true,
                       "pred_oracle_minus_true_z4": row5.contrast_z_oracle_minus_true})
    D2 = pd.DataFrame(d2)
    D2["first_order_resid_z4"] = D2.items_effect_z4 - D2.pred_items_effect_z4
    D2.to_csv(out / "D2_decomposition.csv", index=False)
    D2agg = D2.groupby("param", sort=False).agg(
        mean_z_step4=("z_step4", "mean"), mean_z_pilot=("z_pilot", "mean"), mean_z_psi1=("z_psi1", "mean"),
        mean_z_psi2=("z_psi2", "mean"), mean_items_effect_z4=("items_effect_z4", "mean"),
        mean_pred_items_effect_z4=("pred_items_effect_z4", "mean"),
        max_abs_first_order_resid_z4=("first_order_resid_z4", lambda v: np.max(np.abs(v))),
        mean_pred_assumption_z4=("pred_assumption_z4", "mean"), mean_pred_sampling_z4=("pred_sampling_z4", "mean"),
        mean_s1b_effect_z2=("s1b_effect_z2", "mean"), min_s1b_effect_z2=("s1b_effect_z2", "min"),
        max_s1b_effect_z2=("s1b_effect_z2", "max")).reset_index()
    D2agg.to_csv(out / "D2_summary.csv", index=False)

    # ---------------- D3 -------------------------------------------------------------------------
    q_lo, q_hi = np.sqrt(4 / chi2.ppf(0.975, 4)), np.sqrt(4 / chi2.ppf(0.025, 4))   # SD CI factors, n = 5
    d3, d3agg = [], []
    for stage in ("step4", "pilot", "psi1", "psi2"):
        for j, nm in enumerate(names):
            e = np.array([est[stage][r][0][j] - truth[j] for r in range(5)])
            se = np.array([est[stage][r][1][j] for r in range(5)])
            z = e / se
            for r in range(5):
                d3.append({"stage": stage, "param": nm, "dataset": r, "estimate": est[stage][r][0][j], "se": se[r],
                           "err": e[r], "z": z[r], "covered": bool(abs(z[r]) <= Z95)})
            k = int(np.sum(np.abs(z) <= Z95))
            sd = e.std(ddof=1)
            ratio = sd / se.mean()
            d3agg.append({"stage": stage, "param": nm, "covered": k, "covered_CP95": cp_interval(k, 5),
                          "mean_err": e.mean(), "mean_se": se.mean(), "mean_err_over_mean_se": e.mean() / se.mean(),
                          "mean_z": z.mean(), "rms_z": np.sqrt(np.mean(z ** 2)),
                          "sd_err": sd, "sd_over_mean_se": ratio, "sd_over_mean_se_CI95": (ratio * q_lo, ratio * q_hi),
                          "bias_t": e.mean() / (sd / np.sqrt(5)),
                          "centred_coverage_diag": int(np.sum(np.abs(z - z.mean()) <= Z95)),
                          "se_inflation_needed_4of5": float(np.sort(np.abs(z))[3] / Z95)})
    D3 = pd.DataFrame(d3)
    D3a = pd.DataFrame(d3agg)
    D3.to_csv(out / "D3_estimates_long.csv", index=False)
    D3a.to_csv(out / "D3_summary.csv", index=False)
    ratio = pd.DataFrame([{"dataset": r, "param": nm, "se_psi2_over_se_step4": est["psi2"][r][1][j] / est["step4"][r][1][j],
                           "se_psi1_over_se_step4": est["psi1"][r][1][j] / est["step4"][r][1][j]}
                          for r in range(5) for j, nm in enumerate(names)])
    ratio.to_csv(out / "D3_se_ratios.csv", index=False)

    summary = {
        "label": prov["label"], "provenance": prov, "names": names, "truth": truth,
        "D1_affine_by_set": D1.groupby(["item_set", "subscore"])[["c_tau", "c_a", "c_joint", "d_joint", "tau_rmse_raw",
                                                                    "tau_rmse_after", "loga_rmse_raw", "loga_rmse_after",
                                                                    "a_err_mean"]].agg(["min", "max", "mean"])
        .round(4).pipe(lambda t: {f"{i[0]}_{i[1]}": {f"{c[0]}_{c[1]}": v for c, v in row.items()}
                                  for i, row in t.iterrows()}),
        "D1_failed_params": D1agg[D1agg.param.isin(FAILED)].round(3).to_dict(orient="records"),
        "D2_failed_params": D2agg[D2agg.param.isin(FAILED)].round(3).to_dict(orient="records"),
        "D3_failed_params": D3a[(D3a.param.isin(FAILED))].round(3).to_dict(orient="records"),
        "D3_se_ratio_psi2_over_step4": [float(ratio.se_psi2_over_se_step4.min()), float(ratio.se_psi2_over_se_step4.max())],
        "D3_sd_ci_factors_n5": [float(q_lo), float(q_hi)],
        "definitions": {
            "affine": "θ' = (θ − d)/c  ⇒  â = c·a, τ̂ = (τ − d)/c; ψ' = ψ re-expressed on θ' (β0' = (β0 − d)/c, other "
                      "mean terms /c, D and Σε scaled by diag(1/c)); joint (c, d) by unweighted least squares on "
                      "log â − log a − log c and τ̂ − (τ − d)/c",
            "pred_z": "(ψ'(c, d) − ψ_true)/SE, SE of the corresponding fit (ψ̂(1) for N01, ψ̂(2) for S1-b, Step 4 for oracle)",
            "explained_share": "1 − Σ_r (obs_z − pred_z)² / Σ_r obs_z² over the 5 datasets (descriptive; can be negative)",
            "centred_coverage_diag": "# datasets with |z − mean z| ≤ 1.96 — a diagnostic of dispersion around the bias, "
                                     "NOT a valid confidence interval and NOT a re-scoring of Step 6",
            "se_inflation_needed_4of5": "smallest common factor k with |z|/k ≤ 1.96 in ≥ 4 of 5 datasets = (4th smallest "
                                        "|z|)/1.96; ignores that a real SE correction would also change by dataset",
            "sd_over_mean_se_CI95": "sample SD of errors / mean SE, with the χ²(4) 95 % interval for the SD (×0.599 to ×2.874)",
            "covered_CP95": "Clopper–Pearson 95 % interval for the coverage proportion k/5"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=conv))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    print("D1-D3 written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
