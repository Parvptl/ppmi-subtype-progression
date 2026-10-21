"""D1 observed-cohort replicate-label diagnostic (approved documented deviation).

Specification: docs/cluster_phase/15_D1_REPLICATE_LABEL_DIAGNOSTIC_REVIEW.md section 5.
Approved 2026-10-16 subject to its exact-match safeguards. Not a D1 result: the
D1 verdict, threshold, power and statistic definitions are read-only and are
never recomputed or rewritten.

Order, each step gated on the previous one:
  0  provenance.json written before any computation; refuse on any mismatch
     (code hashes vs 00_protocol.json, panel MD5, folds, pip freeze, threads, host)
  1  null cohort 000 re-executed exactly as in the frozen run; must equal
     cohorts/null/null_000.json exactly, otherwise stop (observed never touched)
  2  observed stage re-executed with the label recorder; must equal
     05_observed.json exactly, otherwise labels are quarantined and not analysed
  3  labels saved; 4  exploratory, uncalibrated splits (doc 13 section 5 D-a, D-b);
  5  frozen checksums re-verified.

    scripts/cluster/submit.sh d1_observed_replicate_labels_DIAGNOSTIC \
        --partition=normal -w node004 --time=01:00:00 -- \
        scripts/43_d1_replicate_label_diagnostic.py --out @RUN_DIR@/results
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import loaders  # noqa: E402
from ppmi_subtype import ppc_diagnostic as D  # noqa: E402
from ppmi_subtype import raw_scores as RS  # noqa: E402
from ppmi_subtype import replicate_label_diagnostic as RL  # noqa: E402
from ppmi_subtype import stage1_checks as S  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN_RUN = ROOT / "experiments/20261016T070746Z_d1_full"
FROZEN = FROZEN_RUN / "results"
SPEC = "docs/cluster_phase/15_D1_REPLICATE_LABEL_DIAGNOSTIC_REVIEW.md"
NEW_CODE = ["scripts/43_d1_replicate_label_diagnostic.py",
            "src/ppmi_subtype/replicate_label_diagnostic.py", "src/ppmi_subtype/stage1_checks.py",
            "src/ppmi_subtype/raw_scores.py"]
REQUIRED_HOST = "node004"
SEEDS = {"null_000": [42, 41, 0, 0], "observed": [42, 41, 9, 0], "run_cohort_suffix": [7],
         "fits": "SeedSequence(entropy + [7, fold, 1]) first word mod 2^31",
         "replicates": "default_rng(entropy + [7, fold, 2])"}
P_STRATA = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 99))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def frozen_manifest() -> dict:
    """SHA-256 of every file under the frozen results directory."""
    return {str(p.relative_to(FROZEN)): sha(p) for p in sorted(FROZEN.rglob("*")) if p.is_file()}


def verify_frozen() -> dict:
    """Locks, the 04_outputs cohort checksums, generator npz and folds."""
    bad = []
    locks = sorted(FROZEN.glob("*.lock"))
    for lk in locks:
        want = json.loads(lk.read_text())["sha256"]
        if sha(FROZEN / f"{lk.stem}.json") != want:
            bad.append(lk.stem)
    outs = json.loads((FROZEN / "04_outputs.json").read_text())["sha256"]
    bad += [f for f, h in outs.items() if sha(FROZEN / f) != h]
    if sha(FROZEN / "fullcohort_tt_draws.npz") != json.loads((FROZEN / "00_generator.json").read_text())["npz_sha256"]:
        bad.append("fullcohort_tt_draws.npz")
    prot = json.loads((FROZEN / "00_protocol.json").read_text())["hashes"]
    if sha(FROZEN / "folds.csv") != prot["folds.csv"]:
        bad.append("folds.csv")
    return {"locks_checked": len(locks), "cohort_files_checked": len(outs), "failures": bad,
            "ok": not bad}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json))


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def provenance(out: Path, allow_host: bool) -> tuple[dict, list[str]]:
    prot = json.loads((FROZEN / "00_protocol.json").read_text())
    problems = []
    code = {}
    for f, h in prot["hashes"].items():
        if f.endswith(".py"):
            cur = sha(ROOT / f)
            code[f] = {"frozen": h, "current": cur, "same": cur == h}
            if cur != h:
                problems.append(f"code hash differs: {f}")
    cfg = load_config()
    panel_md5 = sha(cfg.path("outputs") / "visit_panel.csv", "md5")
    if panel_md5 != prot["hashes"]["panel_md5"]:
        problems.append("panel MD5 differs")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    frozen_freeze = (FROZEN_RUN / "pip_freeze.txt").read_text()
    if freeze != frozen_freeze:
        problems.append("pip freeze differs from the frozen run")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    if any(x != "1" for x in threads.values()):
        problems.append(f"thread settings not single-threaded: {threads}")
    host = platform.node()
    if host != REQUIRED_HOST and not allow_host:
        problems.append(f"host {host} is not {REQUIRED_HOST}")
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except OSError:
        lscpu = ""
    cpu = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")), "?")
    import numpy as _np
    import scipy as _sp
    prov = {
        "purpose": "descriptive diagnostic, NOT a D1 result (doc 15; approved documented deviation)",
        "spec": SPEC, "spec_sha256": sha(ROOT / SPEC),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host, "required_host": REQUIRED_HOST, "allow_host_override": allow_host,
        "cpu_model": cpu, "lscpu": lscpu, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "threads": threads, "python": sys.version, "numpy": _np.__version__, "scipy": _sp.__version__,
        "pip_freeze_identical_to_frozen_run": freeze == frozen_freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "d1_code_hashes": code, "new_code_sha256": {f: sha(ROOT / f) for f in NEW_CODE},
        "panel_md5": panel_md5, "folds_sha256": sha(FROZEN / "folds.csv"),
        "frozen_05_observed_sha256": sha(FROZEN / "05_observed.json"),
        "frozen_null_000_sha256": sha(FROZEN / "cohorts/null/null_000.json"),
        "frozen_protocol_lock": json.loads((FROZEN / "00_protocol.lock").read_text())["sha256"],
        "seeds": SEEDS, "R": prot["R"], "fit": prot["fit"], "nu_prior": prot["nu_prior"],
        "sampler": prot["sampler"], "problems": problems,
    }
    return prov, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-host", action="store_true",
                    help="testing only: skip the node004 requirement (recorded in provenance)")
    ap.add_argument("--provenance-only", action="store_true",
                    help="write provenance and verify the frozen run, then stop (no computation)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    status = {"steps": {}}

    def finish(code, msg):
        status.update({"exit": code, "message": msg,
                       "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        after = verify_frozen()
        status["frozen_after"] = after
        status["frozen_manifest_unchanged"] = frozen_manifest() == before_manifest
        write(out / "status.json", status)
        log(f"{msg} (exit {code}); frozen after: ok={after['ok']}, "
            f"manifest unchanged={status['frozen_manifest_unchanged']}")
        return code

    # ---- 0. provenance and frozen checks, before any computation -------------------------
    before_manifest = frozen_manifest()
    write(out / "frozen_manifest_before.json", before_manifest)
    before = verify_frozen()
    prov, problems = provenance(out, a.allow_host)
    prov["frozen_before"] = before
    write(out / "provenance.json", prov)
    (out / "README.md").write_text(
        "# D1 observed-cohort replicate labels: DIAGNOSTIC\n\n"
        "Descriptive diagnostic under an approved, documented protocol deviation\n"
        f"(`{SPEC}`). **Not a D1 result.** The D1 verdict (NOT REPRODUCED), threshold,\n"
        "power and statistic definitions are unchanged and were only read. Every split\n"
        "here is exploratory and uncalibrated: no null calibration, no threshold, no verdict.\n"
        "See status.json for the gate results.\n")
    status["steps"]["0_provenance"] = {"problems": problems, "frozen_before": before}
    if problems or not before["ok"]:
        return finish(2, f"provenance/frozen check failed: {problems or before['failures']}")
    log(f"provenance written; host {prov['host']}, cpu {prov['cpu_model']}")
    if a.provenance_only:
        return finish(0, "provenance only: no computation run")

    cfg = load_config()
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    data = hbayes.build_growth_data(panel)
    folds = D.make_folds(data, 5, 42)
    if hashlib.sha256(folds.to_csv(index=False).encode()).hexdigest() != prov["folds_sha256"]:
        return finish(2, "rebuilt folds differ from the frozen folds.csv")
    prot = json.loads((FROZEN / "00_protocol.json").read_text())
    R, fit_kw = prot["R"], prot["fit"]
    z = np.load(FROZEN / "fullcohort_tt_draws.npz")
    fullpost = types.SimpleNamespace(**{k: z[k] for k in z.files}, n_draws=len(z["sigma2"]))

    # ---- 1. determinism pre-check: null cohort 000 -----------------------------------
    t0 = time.time()
    ent = SEEDS["null_000"]
    rng = np.random.default_rng(ent)
    theta = D.theta_from(fullpost, int(rng.integers(fullpost.n_draws)))
    y0, inside = D.generate(data, theta, rng, None)
    res0 = D.run_cohort(data, y0, folds, R, fit_kw, list(ent) + SEEDS["run_cohort_suffix"])
    res0["n_visits_in_episode"] = int(inside.sum())
    frozen0 = json.loads((FROZEN / "cohorts/null/null_000.json").read_text())
    g1 = RL.gate(res0, frozen0, RL.GATED_KEYS_NULL + ("n_visits_in_episode",))
    g1["runtime_s"] = time.time() - t0
    status["steps"]["1_null_000_precheck"] = g1
    write(out / "gate1_null_000.json", g1)
    log(f"gate 1 (null 000): passed={g1['passed']}, {len(g1['mismatches'])} mismatches, "
        f"{g1['runtime_s']:.0f}s")
    if not g1["passed"]:
        return finish(3, "null-cohort 000 pre-check failed: observed stage NOT executed")

    # ---- 2. observed stage with the label recorder ---------------------------------
    t0 = time.time()
    with RL.LabelRecorder() as rec:
        res = D.run_cohort(data, data.y, folds, R, fit_kw, SEEDS["observed"] + SEEDS["run_cohort_suffix"])
    Lrep = rec.single()
    frozen_obs = json.loads((FROZEN / "05_observed.json").read_text())["observed"]
    g2 = RL.gate(res, frozen_obs, RL.GATED_KEYS)
    # the recorded labels must regenerate the gated replicate means exactly
    S_rep = D.label_statistics(Lrep, data.patient_index)
    names = D.STAT_NAMES + D.DESC_NAMES
    rm = dict(zip(names, np.nanmean(S_rep, axis=0).tolist()))
    g2["recorded_labels_regenerate_S_rep_mean"] = not RL.exact_mismatches(
        json.loads(json.dumps(rm)), frozen_obs["S_rep_mean"])
    g2["passed"] = g2["passed"] and g2["recorded_labels_regenerate_S_rep_mean"]
    g2["runtime_s"] = time.time() - t0
    status["steps"]["2_observed_reexecution"] = g2
    write(out / "gate2_observed.json", g2)
    log(f"gate 2 (observed): passed={g2['passed']}, {len(g2['mismatches'])} mismatches, "
        f"{g2['runtime_s']:.0f}s")
    meta = pd.DataFrame({"PATNO": data.patients[data.patient_index], "t": data.t,
                         "fold": folds.set_index("PATNO").fold.reindex(data.patients).to_numpy()[data.patient_index]})
    if not g2["passed"]:
        q = out / "REJECTED_nonidentical"
        q.mkdir()
        np.savez_compressed(q / "labels_REJECTED.npz", Lrep=Lrep)
        (q / "README.txt").write_text("Observed re-execution did NOT match 05_observed.json exactly.\n"
                                      "These labels are quarantined and must not be analysed.\n")
        return finish(4, "observed re-execution not identical: labels quarantined, not analysed")

    # ---- 3. save labels ---------------------------------------------------------------
    np.savez_compressed(out / "labels.npz", Lrep=Lrep, PATNO=meta.PATNO.to_numpy(),
                        t=meta.t.to_numpy(), fold=meta.fold.to_numpy())
    status["steps"]["3_labels"] = {"file": "labels.npz", "sha256": sha(out / "labels.npz"),
                                   "shape": list(Lrep.shape), "dtype": str(Lrep.dtype)}

    # ---- 4. exploratory, uncalibrated splits -------------------------------------------
    part2, part3 = loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg)
    frame = RS.build_visit_frame(panel, RS.item_visit_table(part2, part3, cfg.subtype), cfg.subtype)
    if not RS.validate_reconstruction(frame)["exact"]:
        return finish(5, "item reconstruction mismatch: splits not computed")
    v = frame[frame.ratio.notna() & np.isfinite(frame.ratio) & frame.PATNO.isin(data.patients)]
    v = v.sort_values(["PATNO", "years_from_baseline"]).reset_index(drop=True)
    assert np.array_equal(v.ratio.to_numpy(), data.y) and np.array_equal(v.years_from_baseline.to_numpy(), data.t)
    P_ = v.pigd_sum.to_numpy(int)
    strata = np.full(len(P_), -1)
    for k, (_, lo, hi) in enumerate(P_STRATA):
        strata[(P_ >= lo) & (P_ <= hi)] = k
    Lobs = D.labels_from_ratio(data.y)
    pid = data.patient_index
    sh_obs = RL.stratum_shares(Lobs, strata, len(P_STRATA))[0]
    sh_rep = RL.stratum_shares(Lrep, strata, len(P_STRATA))
    rows1 = []
    for k, (name, _, _) in enumerate(P_STRATA):
        sel = strata == k
        rows1.append({"pigd_sum": name, "visits": int(sel.sum()),
                      "observed_ind": int((Lobs[sel] == S.IND).sum()),
                      "rep_mean_ind_count": float((Lrep[:, sel] == S.IND).sum(1).mean()),
                      **RL.describe(sh_obs[k], sh_rep[:, k])})
    c_obs = RL.direction_outcome_counts(Lobs, pid)[0]
    c_rep = RL.direction_outcome_counts(Lrep, pid)
    r_obs, r_rep = RL.reversion_rates(c_obs), RL.reversion_rates(c_rep)
    rows2 = []
    for d, dname in enumerate(S.DIRECTIONS):
        row = {"direction": dname, "R_D": RL.describe(r_obs[d], r_rep[:, d])}
        for o, oname in enumerate(S.OUTCOMES):
            row[oname] = RL.describe(c_obs[d, o], c_rep[:, d, o])
        rows2.append(row)
    both_obs = RL.reversion_rates(c_obs.sum(0))
    both_rep = RL.reversion_rates(c_rep.sum(1))
    assert np.isclose(both_obs, frozen_obs["S_obs"]["S3_R_D"])
    rows2.append({"direction": "both", "R_D": RL.describe(both_obs, both_rep)})
    splits = {"status": "exploratory, uncalibrated; no threshold, no verdict, no new hypothesis test",
              "reference": "patient-conditional replicates of the observed D1 stage (R = 500)",
              "midp_convention": "D1: (#rep > obs + 0.5 #= + 0.5)/(R + 1); z < 0 = observed above replicates",
              "ind_by_pigd_sum": rows1, "direction_outcomes": rows2}
    write(out / "splits.json", splits)
    pd.DataFrame([{k: v for k, v in r.items()} for r in rows1]).to_csv(out / "ind_by_pigd_sum.csv", index=False)
    flat = []
    for r in rows2:
        for k, val in r.items():
            if isinstance(val, dict):
                flat.append({"direction": r["direction"], "quantity": k, **val})
    pd.DataFrame(flat).to_csv(out / "direction_outcomes.csv", index=False)
    status["steps"]["4_splits"] = {"files": ["splits.json", "ind_by_pigd_sum.csv", "direction_outcomes.csv"]}
    log("splits written")
    return finish(0, "complete: both gates passed")


if __name__ == "__main__":
    sys.exit(main())
