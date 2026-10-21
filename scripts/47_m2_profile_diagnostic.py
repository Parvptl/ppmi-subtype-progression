"""Diagnostic: Laplace vs D3-reference likelihood slice over the PIGD visit-level SD.

Doc 18 section 6 ("optional cheaper step"). Not a repair, not an amendment.
Dataset 0 of the pilot's recovery (seed [42, 45, 1, 0]), item parameters fixed
at their generating values, design of the real cohort (times, treatment only).

Slice: σ_P (PIGD visit-level SD) varies over 7 pre-specified values; σ_T, the
visit-level tremor-PIGD correlation ρ and all 16 other ψ components are held
at the dataset-0 fixed-item Laplace MLE ψ̂ (from the fixed-item run's
results.json). This is a conditional slice, not a re-maximised profile.

Two log-likelihoods per grid point, summed over the 934 patients:
* Laplace: `m2_grm.group_terms` (the fitted approximation, unchanged);
* reference: the D3 reference exactly as in the pilot's Laplace check:
  `m2_grm.agq_one` (7 nodes) when 2n <= 6, else `m2_grm.is_one`
  (20,000 draws, t5 proposal at the Laplace mode). Common random numbers:
  the importance draws of patient i use seed [42, 47, s, i] at every grid
  point; two independent seed sets s = 0, 1 assess Monte Carlo stability.

Comparison rules (fixed in config.json before computation): see PRESPEC.
No gate, label or evaluation statistic is computed.

    scripts/cluster/submit.sh stage2_m2_profile_diag --partition=normal --time=01:00:00 -- \
        scripts/47_m2_profile_diagnostic.py --out @RUN_DIR@/results --jobs 32
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SPEC = ROOT / "docs/cluster_phase/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md"
SPEC_LOCK = ROOT / "docs/cluster_phase/_frozen/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256"
PILOT = ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results"
FIXED = ROOT / "experiments/20261016T203341Z_stage2_m2_recovery_fixed_items_ds0to4"
GRID = [0.3550, 0.3675, 0.3795, 0.3925, 0.4053, 0.4200, 0.4350]
PRESPEC = {
    "parameter": "sigma_P = sqrt(Sigma_e[1,1]), PIGD visit-level SD on the stage-1 scale",
    "grid": GRID,
    "grid_notes": "0.3795 = dataset-0 fixed-item Laplace MLE; 0.4053 = truth; spacing ~1 Wald SE (0.0134)",
    "nuisance": "sigma_T, rho (visit-level tremor-PIGD correlation) and the 16 other psi components held "
                "at the dataset-0 fixed-item Laplace MLE (conditional slice, no re-maximisation)",
    "laplace": "m2_grm.group_terms total over patients",
    "reference": "D3: agq_one(nodes=7) if 2n <= 6 else is_one(draws=20000, df=5); CRN seeds [42,47,s,i], s in {0,1}",
    "argmax": "vertex of a least-squares quadratic in sigma_P through the 7 points (grid argmax also reported)",
    "material": "|argmax_reference - argmax_laplace| > 0.0134 (dataset-0 Wald SE of sigma_P, delta method)",
    "supports_mechanism_if": "material AND reference argmax closer to the truth (0.4053) than the Laplace argmax "
                             "AND (Laplace - reference) decreases with sigma_P across the grid, for both seed sets",
    "self_check": "Laplace slice at sigma_P = MLE equals -nll of the fixed-item fit to 1e-6 relative",
    "not_computed": "fidelity gates, labels, prevalence, transitions, calibration, verdicts",
}
_W: dict = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_script(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def psi_at_sigma_p(psi_hat, sigma_p):
    """ψ̂ with Σ_ε[1,1] = σ_P², keeping σ_T and ρ at their ψ̂ values."""
    beta, D, _, S, _ = M.unpack(psi_hat)
    sT, sP = np.sqrt(S[0, 0]), np.sqrt(S[1, 1])
    rho = S[0, 1] / (sT * sP)
    S2 = np.array([[sT ** 2, rho * sT * sigma_p], [rho * sT * sigma_p, sigma_p ** 2]])
    return M.to_psi(beta, D, S2)


def _ref(task):
    i, s, k = task
    t, x, YT, YP = _W["patients"][i]
    psi, it = _W["psis"][k], _W["it"]
    if 2 * len(t) <= 6:
        return i, s, k, M.agq_one(t, x, YT, YP, psi, it, nodes=7), 0.0
    est, se = M.is_one(t, x, YT, YP, psi, it, np.random.default_rng([42, 47, s, i]), draws=20000)
    return i, s, k, est, se


def quad_vertex(x, y):
    c = np.polyfit(x, y, 2)
    return float(-c[1] / (2 * c[0])) if c[0] < 0 else float("nan"), c.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()
    pilot_prov = json.loads((PILOT / "provenance.json").read_text())
    fixed_prov = json.loads((FIXED / "provenance.json").read_text())
    config = {"prespecification": PRESPEC, "spec_sha256": sha(SPEC), "spec_frozen": SPEC_LOCK.read_text().split()[0],
              "pilot_code_identical": {f: sha(ROOT / f) == h for f, h in pilot_prov["code_sha256"].items()},
              "script46_identical": sha(ROOT / "scripts/46_m2_recovery_fixed_items.py")
              == fixed_prov["new_code_sha256"]["scripts/46_m2_recovery_fixed_items.py"],
              "script47_sha256": sha(ROOT / "scripts/47_m2_profile_diagnostic.py"),
              "fixed_results_sha256": sha(FIXED / "results.json"), "host": platform.node(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "jobs": a.jobs}
    (out / "config.json").write_text(json.dumps(config, indent=2))       # before any computation
    if (config["spec_sha256"] != config["spec_frozen"] or not all(config["pilot_code_identical"].values())
            or not config["script46_identical"]):
        print("hash mismatch; nothing computed", file=sys.stderr)
        return 2

    P = load_script("m2_pilot", "45_m2_pilot.py")
    cfg = load_config()
    v, times, treat, YTd, YPd, *_ = P.load_cohort(cfg)
    groups = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, _ = P.truth_on_stage1_scale(groups)
    fixed = json.loads((FIXED / "results.json").read_text())
    assert np.array_equal(np.asarray(fixed["psi_true"]), psi_true)
    d0 = fixed["datasets"][0]
    assert d0["dataset"] == 0
    psi_hat = np.asarray(d0["psi_hat"])
    sg = P.regroup(groups, *P.simulate_groups(np.random.default_rng([42, 45, 1, 0]), groups, psi_true, it_true))

    sig_true = float(np.sqrt(M.unpack(psi_true)[3][1, 1]))
    sig_hat = float(np.sqrt(M.unpack(psi_hat)[3][1, 1]))
    psis = [psi_at_sigma_p(psi_hat, s) for s in GRID]
    # Laplace slice (vectorised, exact as fitted)
    lap = [-M.negloglik(p, sg, it_true, want_grad=False) for p in psis]
    self_check = -M.negloglik(psi_hat, sg, it_true, want_grad=False)
    sc_ok = bool(np.isclose(self_check, -d0["nll"], rtol=1e-6, atol=0))
    if not sc_ok:
        (out / "results.json").write_text(json.dumps({"self_check_failed": [self_check, -d0["nll"]]}))
        print("self-check failed: dataset or ψ̂ differs from the fixed-item run", file=sys.stderr)
        return 3
    print(f"[{time.strftime('%H:%M:%S')}] Laplace slice done; self-check ok", flush=True)

    patients = []
    for n, g in sg.items():
        for r in range(g.T.shape[0]):
            patients.append((g.T[r], g.X[r], g.YT[r], g.YP[r]))
    _W.update(patients=patients, psis=psis, it=it_true)
    tasks = []
    for i, (t, *_rest) in enumerate(patients):
        for k in range(len(GRID)):
            tasks.append((i, 0, k))
            if 2 * len(t) > 6:
                tasks.append((i, 1, k))                  # AGQ is deterministic: one evaluation suffices
    ref = np.zeros((2, len(GRID), len(patients)))
    var = np.zeros((2, len(GRID)))
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, initializer=_W.update, initargs=(dict(_W),)) as ex:
        for done, (i, s, k, est, se) in enumerate(ex.map(_ref, tasks, chunksize=8)):
            ref[s, k, i] = est
            var[s, k] += se ** 2
            if 2 * len(patients[i][0]) <= 6:
                ref[1, k, i] = est
            if done % 2000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] reference {done}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    tot = ref.sum(2)
    lap = np.asarray(lap)
    rows = []
    for k, s in enumerate(GRID):
        rows.append({"sigma_P": s, "laplace": lap[k], "reference_s0": tot[0, k], "reference_s1": tot[1, k],
                     "ref_mc_se_s0": float(np.sqrt(var[0, k])), "ref_mc_se_s1": float(np.sqrt(var[1, k])),
                     "gap_s0": lap[k] - tot[0, k], "gap_s1": lap[k] - tot[1, k]})
    vl, cl = quad_vertex(GRID, lap)
    v0, c0 = quad_vertex(GRID, tot[0])
    v1, c1 = quad_vertex(GRID, tot[1])
    se_sd = 0.0134
    gaps = [np.array([r[f"gap_s{s}"] for r in rows]) for s in (0, 1)]
    decreasing = [bool(np.all(np.diff(g) < 0)) for g in gaps]
    material = [abs(v - vl) > se_sd for v in (v0, v1)]
    closer = [abs(v - sig_true) < abs(vl - sig_true) for v in (v0, v1)]
    res = {"rows": rows, "sigma_true": sig_true, "sigma_hat_laplace_mle": sig_hat,
           "argmax_quadratic": {"laplace": vl, "reference_s0": v0, "reference_s1": v1},
           "argmax_grid": {"laplace": GRID[int(np.argmax(lap))], "reference_s0": GRID[int(np.argmax(tot[0]))],
                           "reference_s1": GRID[int(np.argmax(tot[1]))]},
           "quadratic_coefs": {"laplace": cl, "reference_s0": c0, "reference_s1": c1},
           "material": material, "reference_closer_to_truth": closer, "gap_decreasing_in_sigma": decreasing,
           "supports_mechanism": [m and c and dcr for m, c, dcr in zip(material, closer, decreasing)],
           "self_check": {"laplace_at_mle": self_check, "fixed_fit_minus_nll": -d0["nll"], "ok": sc_ok},
           "n_patients": len(patients), "n_tasks": len(tasks),
           "n_agq_patients": int(sum(2 * len(p[0]) <= 6 for p in patients)),
           "runtime_s": time.time() - t_start, "reference_runtime_s": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("argmax_quadratic", "argmax_grid", "material",
                                          "reference_closer_to_truth", "gap_decreasing_in_sigma",
                                          "supports_mechanism")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
