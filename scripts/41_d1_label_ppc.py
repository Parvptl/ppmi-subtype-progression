"""D1 posterior-predictive label check: smoke, mini-calibration and the full staged run.

Protocol: docs/cluster_phase/12_CONSOLIDATED_RESEARCH_REPORT.md Appendix B
(B.10 Revision 2; B.11 Amendment 3; B.12 Amendment 4). The full 500-null /
positive-control / observed run is deliberately NOT implemented here; it
needs explicit approval after the smoke results are reviewed.

    python scripts/41_d1_label_ppc.py --stage smoke --out <new dir> [--jobs 16]
    python scripts/41_d1_label_ppc.py --stage mini  --out <new dir> [--jobs 32]
    python scripts/41_d1_label_ppc.py --stage full  --out <dir> [--jobs 32] [--resume]

smoke: 3 null + 1 PC4 + 1 PC4v (paired) + the observed cohort (timing only),
       R = 50, short chains; plus 50 generation-only paired cohorts for the
       PC4/PC4v/null direction check. No threshold, no verdict.
mini:  20 null cohorts at full settings (R = 500, frozen chains), tag 90.
       Reports timing, p bounds, cap frequency, null-T summary. These T are
       NOT a threshold and are never reused in the full run.
The observed cohort's statistics, z and T are never written by this script.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import hbayes_tt_nu as NU  # noqa: E402
from ppmi_subtype import ppc_diagnostic as D  # noqa: E402
from ppmi_subtype import ppc_runner as RUN  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
REPORT = "docs/cluster_phase/12_CONSOLIDATED_RESEARCH_REPORT.md"
CODE = ["src/ppmi_subtype/ppc_diagnostic.py", "src/ppmi_subtype/ppc_runner.py",
        "scripts/41_d1_label_ppc.py",
        "src/ppmi_subtype/hbayes_tt_nu.py", "src/ppmi_subtype/hbayes_tt.py",
        "src/ppmi_subtype/hbayes_t.py", "src/ppmi_subtype/hbayes.py",
        "src/ppmi_subtype/indeterminate_audit.py", "src/ppmi_subtype/subtype.py"]
PC4 = {"pi": 0.30, "L": 1.02, "D": 2.0, "kappa": 0.25, "replace_level": True}
PC4V = dict(PC4, replace_level=False)
SMOKE_FIT = {"n_draws": 400, "burn_in": 200, "thin": 1}
_S: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _cohort(task):
    kind, entropy = task
    d, fullpost, folds, R, fit_kw = _S["data"], _S["post"], _S["folds"], _S["R"], _S["fit_kw"]
    t0 = time.time()
    if kind == "observed":
        y, inside = d.y, None
    else:
        rng = np.random.default_rng(entropy)
        theta = D.theta_from(fullpost, int(rng.integers(fullpost.n_draws)))
        ep = {"null": None, "PC4": PC4, "PC4v": PC4V}[kind]
        y, inside = D.generate(d, theta, rng, ep)
    res = D.run_cohort(d, y, folds, R, fit_kw, list(entropy) + [7],
                       record_stats=(kind != "observed"))
    res.update({"kind": kind, "entropy": list(entropy), "wall_s": time.time() - t0})
    if inside is not None:
        res["n_visits_in_episode"] = int(inside.sum())
    return res


def _generation_only(entropy):
    """Label statistics of generated cohorts (no fitting): null vs PC4 vs PC4v, paired."""
    d, fullpost = _S["data"], _S["post"]
    out = {}
    for kind, ep in (("null", None), ("PC4", PC4), ("PC4v", PC4V)):
        rng = np.random.default_rng(entropy)
        theta = D.theta_from(fullpost, int(rng.integers(fullpost.n_draws)))
        y, _ = D.generate(d, theta, rng, ep)
        S = D.label_statistics(D.labels_from_ratio(y)[None, :], d.patient_index)[0]
        out[kind] = S.tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "mini", "full"), required=True)
    ap.add_argument("--resume", action="store_true", help="full stage: continue an existing run")
    ap.add_argument("--codepath-check", action="store_true",
                    help="full stage wiring check only: N_null=19, N_pc=3, R=20, short chains")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "full":
        return full(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if sha(panel_path, "md5") != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    panel = pd.read_csv(panel_path, low_memory=False)
    data = hbayes.build_growth_data(panel)
    folds = D.make_folds(data, 5, 42)
    folds.to_csv(out / "folds.csv", index=False)
    smoke = a.stage == "smoke"
    R = 50 if smoke else 500
    fit_kw = SMOKE_FIT if smoke else D.FIT_KW
    manifest = {
        "stage": a.stage, "protocol": f"{REPORT} Appendix B.10-B.12",
        "sha256": {f: sha(ROOT / f) for f in [REPORT] + CODE},
        "panel_md5": PANEL_MD5, "fold_file_sha256": sha(out / "folds.csv"),
        "cohort": {"patients": data.n_patients, "ratio_defined_visits": data.n_obs},
        "R": R, "fit": fit_kw, "nu_prior": "gamma2_true (truncated Gamma(2, 0.1))",
        "sampler": "collapsed_slice", "seed_tag": 91 if smoke else 90,
        "PC4": PC4, "PC4v": PC4V,
        "not_produced_here": ["threshold c", "P_lo", "observed T/z/statistics", "verdict"],
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"manifest written (data: {data.n_patients} patients, {data.n_obs} visits)")

    tf = time.time()
    fullpost = NU.fit(data, "TT", nu_update="collapsed_slice",
                      log_prior_nu=NU.log_prior_nu_gamma2_true, seed=42, **fit_kw)
    np.savez_compressed(out / "fullcohort_tt_draws.npz", beta=fullpost.beta,
                        sigma2=fullpost.sigma2, tau0_2=fullpost.tau0_2, tau1_2=fullpost.tau1_2,
                        nu=fullpost.nu, nu_e=fullpost.nu_e)
    manifest["artifacts"] = {"fullcohort_tt_draws.npz": sha(out / "fullcohort_tt_draws.npz"),
                             "fullcohort_fit_s": round(time.time() - tf, 1)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    log("full-cohort TT fit (null generator) done")

    tag = 91 if smoke else 90
    if smoke:
        tasks = [("null", [42, 41, tag, c]) for c in range(3)] + \
                [("PC4", [42, 41, tag, 10]), ("PC4v", [42, 41, tag, 10]),
                 ("observed", [42, 41, tag, 99])]
    else:
        tasks = [("null", [42, 41, tag, c]) for c in range(20)]
    state = {"data": data, "post": fullpost, "folds": folds, "R": R, "fit_kw": fit_kw}
    with ProcessPoolExecutor(min(a.jobs, len(tasks)), initializer=_S.update,
                             initargs=(state,)) as ex:
        res = list(ex.map(_cohort, tasks))
        gen = list(ex.map(_generation_only, [[42, 41, tag, 1000 + c] for c in range(50)])) \
            if smoke else []
    log("cohorts done")

    cap = D.z_cap(R)
    rows = []
    for r in res:
        row = {"kind": r["kind"], "wall_s": round(r["wall_s"], 1),
               **{k: round(v, 1) for k, v in r["timing"].items()}}
        if r["kind"] != "observed":           # observed cohort: timing only (B.12.3)
            row.update({"p_at_bound": r["p_at_bound"], "T_at_cap": r["T_at_cap"], "T": r["T"], **{f"z_{k}": v for k, v in r["z"].items()},
                        **{f"obs_{k}": v for k, v in r["S_obs"].items()},
                        "n_visits_in_episode": r.get("n_visits_in_episode")})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"{a.stage}_cohorts.csv", index=False)
    summary = {"stage": a.stage, "R": R, "z_cap": cap, "cohorts": len(res),
               "wall_per_cohort_s": tab.wall_s.describe().round(1).to_dict(),
               "fit_s_per_cohort": tab.fit_s.describe().round(1).to_dict(),
               "replicate_s_per_cohort": tab.replicate_s.describe().round(1).to_dict(),
               "stats_s_per_cohort": tab.stats_s.describe().round(1).to_dict(),
               "share_of_p_at_bounds_synthetic": float(tab.p_at_bound.sum() /
                                                       (5 * tab.p_at_bound.notna().sum())),
               "share_T_at_cap_synthetic": float(tab.T_at_cap.dropna().astype(bool).mean()),
               "n_rep_undefined_total_synthetic": {
                   k: int(sum(r["n_rep_undefined"][k] for r in res if "n_rep_undefined" in r))
                   for k in D.STAT_NAMES + D.DESC_NAMES}}
    nul = tab[tab.kind == "null"]
    summary["null_T"] = {"values": nul["T"].round(4).tolist(),
                         "note": "NOT a threshold; not reused in the full run"}
    if smoke:
        g = {k: np.array([x[k] for x in gen]) for k in ("null", "PC4", "PC4v")}
        names = list(D.STAT_NAMES + D.DESC_NAMES)
        summary["generation_only_direction_check"] = {
            "cohorts": 50,
            "mean_label_statistics": {k: dict(zip(names, np.nanmean(v, axis=0).round(4).tolist()))
                                      for k, v in g.items()},
            "paired_PC4_minus_PC4v_mean": dict(zip(names, np.nanmean(g["PC4"] - g["PC4v"],
                                                                      axis=0).round(4).tolist())),
            "paired_PC4v_minus_null_mean": dict(zip(names, np.nanmean(g["PC4v"] - g["null"],
                                                                       axis=0).round(4).tolist()))}
        summary["pc_signed_z_smoke"] = tab[tab.kind.isin(["PC4", "PC4v"])][
            ["kind", "T"] + [f"z_{k}" for k in D.STAT_NAMES]].round(3).to_dict("records")
    summary["runtime_total_s"] = round(time.time() - t0, 1)
    (out / f"{a.stage}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done in {summary['runtime_total_s']} s")
    return 0


# ---------------------------------------------------------------------------
#  Full staged run
# ---------------------------------------------------------------------------
def _full_cohort(item):
    key, entropy, payload = item
    d, fullpost, folds = _S["data"], _S["post"], _S["folds"]
    t0 = time.time()
    rng = np.random.default_rng(entropy)
    theta = D.theta_from(fullpost, int(rng.integers(fullpost.n_draws)))
    ep = payload.get("episode")
    ep = None if ep is None else {k: v for k, v in ep.items() if k != "tag"}
    y, inside = D.generate(d, theta, rng, ep)
    res = D.run_cohort(d, y, folds, _S["R"], _S["fit_kw"], list(entropy) + [7])
    return {"kind": payload["kind"], "T": res["T"], "z": res["z"], "p": res["p"],
            "S_obs": res["S_obs"], "p_at_bound": res["p_at_bound"], "T_at_cap": res["T_at_cap"],
            "n_rep_undefined": res["n_rep_undefined"], "folds": res["folds"],
            "timing": res["timing"], "n_visits_in_episode": int(inside.sum()),
            "wall_s": time.time() - t0}


def full(a) -> int:
    root = Path(a.out)
    if root.exists() and any(root.iterdir()) and not a.resume:
        print(f"{root} exists and is not empty; use --resume to continue it", file=sys.stderr)
        return 2
    t0 = time.time()
    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if sha(panel_path, "md5") != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    R_, fit_kw = 500, D.FIT_KW
    if a.codepath_check:                          # wiring check only; never a real result
        RUN.N_NULL, RUN.N_PC, R_, fit_kw = 19, 3, 20, SMOKE_FIT
    run = RUN.D1Run(root)
    panel = pd.read_csv(panel_path, low_memory=False)
    data = hbayes.build_growth_data(panel)
    folds = D.make_folds(data, 5, 42)
    fold_bytes = folds.to_csv(index=False).encode()
    if (root / "folds.csv").exists():
        if hashlib.sha256((root / "folds.csv").read_bytes()).hexdigest() != \
                hashlib.sha256(fold_bytes).hexdigest():
            raise RUN.StageError("fold file differs from the frozen one")
    else:
        (root / "folds.csv").write_bytes(fold_bytes)
    protocol = {
        "protocol": f"{REPORT} Appendix B.10-B.14",
        "hashes": {**{f: sha(ROOT / f) for f in [REPORT] + CODE},
                   "panel_md5": PANEL_MD5, "folds.csv": sha(root / "folds.csv")},
        "cohort": {"patients": data.n_patients, "ratio_defined_visits": data.n_obs},
        "codepath_check": a.codepath_check,
        "R": R_, "fit": fit_kw, "nu_prior": "gamma2_true", "sampler": "collapsed_slice",
        "N_null": RUN.N_NULL, "N_pc": RUN.N_PC, "scenarios": RUN.PC_SCENARIOS, "PC4v": RUN.PC4V,
        "primary_power_scenario": RUN.PRIMARY, "power_rule": "two-sided 95% CP lower >= 0.50",
        "threshold_rule": "c = T_(k), k = ceil(0.95 (N+1)), strict T_obs > c",
        "seeds": {"null": "[42, 41, 0, c]", "PC1-PC4": "[42, 41, tag, c]",
                  "PC4v": "[42, 41, 4, c] (paired with PC4)", "observed": RUN.OBSERVED_ENTROPY,
                  "fits/replicates": "SeedSequence(entropy + [7, fold, 1|2])",
                  "generator_fit": 42},
    }
    run.freeze_protocol(protocol)
    log("protocol frozen / verified")
    # null generator: TT fitted to the full ratio-defined cohort (parametric bootstrap)
    gen = root / "fullcohort_tt_draws.npz"
    if run.is_frozen("00_generator"):
        g = run.read_frozen("00_generator")
        if sha(gen) != g["npz_sha256"]:
            raise RUN.StageError("generator draws differ from the frozen file")
    else:
        post = NU.fit(data, "TT", nu_update="collapsed_slice",
                      log_prior_nu=NU.log_prior_nu_gamma2_true, seed=42, **fit_kw)
        np.savez(gen, beta=post.beta, sigma2=post.sigma2, tau0_2=post.tau0_2,
                 tau1_2=post.tau1_2, nu=post.nu, nu_e=post.nu_e)
        run.write_frozen("00_generator", {"npz_sha256": sha(gen), "n_draws": post.n_draws})
    z = np.load(gen)
    import types
    fullpost = types.SimpleNamespace(**{k: z[k] for k in z.files}, n_draws=len(z["sigma2"]))
    state = {"data": data, "post": fullpost, "folds": folds, "R": R_, "fit_kw": fit_kw}

    def observed_fn():
        _S.update(state)
        if a.codepath_check:                      # never evaluate the real observed cohort here
            rng = np.random.default_rng([42, 41, 92, 0])
            y, _ = D.generate(data, D.theta_from(fullpost, 0), rng, None)
            return D.run_cohort(data, y, folds, R_, fit_kw, [42, 41, 92, 0, 7])
        return D.run_cohort(data, data.y, folds, R_, fit_kw, RUN.OBSERVED_ENTROPY + [7])

    status_path = root / "run_status.json"
    try:
        result = RUN.run_all(run, protocol, _full_cohort, observed_fn, jobs=a.jobs, state=state,
                             initializer=_S.update, cap=D.z_cap(R_), log=log)
        status = {"status": "complete", "already_evaluated": result["already_evaluated"]}
    except Exception as e:                       # preserve everything; record the failure
        status = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        status_path.write_text(json.dumps({**status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                           "runtime_s": round(time.time() - t0, 1)}, indent=2))
        log(f"FAILED: {status['error']}")
        return 1
    locks = {p.stem: json.loads(p.read_text())["sha256"] for p in sorted(root.glob("*.lock"))}
    status.update({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "runtime_s": round(time.time() - t0, 1),
                   "lock_checksums": locks, "host": platform.node(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID")})
    status_path.write_text(json.dumps(status, indent=2))
    log(f"complete in {status['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
