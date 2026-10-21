"""Diagnostic: M2 synthetic recovery with item parameters fixed at their generating values.

Not a repair and not a pilot re-run. Doc 17 section 6 option 3: separates the
two candidate causes of the pilot's recovery failure (doc 17 section 4):

* stage 2 itself (the Laplace approximation / longitudinal likelihood), which
  is unchanged when items are known;
* stage 1 (estimated item parameters under the θ ~ N(0, 1) working assumption,
  i.e. the two-stage scale), which is removed when items are known.

Everything else is the pilot's: the same a-priori truth re-expressed on the
stage-1 scale (`truth_on_stage1_scale`), the same real design (934 patients),
the same simulation seeds [42, 45, 1, r] (so the same synthetic datasets), the
same `m2_grm.fit` and Hessian, the same 95 % Wald coverage rule (>= 4 of 5).
Only the item parameters passed to stage 2 differ (truth instead of stage-1
estimates). No observed item data enter the fits; no gate, label or evaluation
statistic is computed.

The pilot's functions are imported from scripts/45_m2_pilot.py, whose SHA-256
must equal the pilot's recorded provenance.

    python scripts/46_m2_recovery_fixed_items.py --out experiments/<UTC>_stage2_m2_recovery_fixed_items \
        --datasets 0          # smallest valid run first
    python scripts/46_m2_recovery_fixed_items.py --out ... --datasets 0 1 2 3 4
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

PILOT = ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results"
SPEC_LOCK = ROOT / "docs/cluster_phase/_frozen/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256"
SPEC = ROOT / "docs/cluster_phase/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pilot_module():
    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", type=int, nargs="+", required=True)
    ap.add_argument("--check-dataset0", action="store_true",
                    help="re-fit stage-1 items on dataset 0 and require the pilot's item-recovery "
                         "summary exactly (proves the synthetic data are the pilot's)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()
    pilot_prov = json.loads((PILOT / "provenance.json").read_text())
    prov = {"purpose": "diagnostic: recovery with item parameters fixed at truth (doc 17 sec. 6 opt. 3); "
                       "not a repair, not a pilot",
            "spec_sha256": sha(SPEC), "spec_frozen": SPEC_LOCK.read_text().split()[0],
            "pilot_dir": str(PILOT.relative_to(ROOT)),
            "code_sha256": {f: sha(ROOT / f) for f in pilot_prov["code_sha256"]},
            "code_identical_to_pilot": {f: sha(ROOT / f) == h for f, h in pilot_prov["code_sha256"].items()},
            "new_code_sha256": {"scripts/46_m2_recovery_fixed_items.py": sha(ROOT / "scripts/46_m2_recovery_fixed_items.py")},
            "datasets": a.datasets, "seeds": "[42, 45, 1, r] (pilot's)", "host": platform.node(),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    if prov["spec_sha256"] != prov["spec_frozen"] or not all(prov["code_identical_to_pilot"].values()):
        print("specification or pilot code differs from the recorded state", file=sys.stderr)
        return 2

    P = load_pilot_module()
    cfg = load_config()
    v, times, treat, YTd, YPd, *_ = P.load_cohort(cfg)          # design only: times, treatment
    groups = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, scale = P.truth_on_stage1_scale(groups)
    pilot_rec = json.loads((PILOT / "step1_recovery.json").read_text())
    assert np.array_equal(np.asarray(pilot_rec["psi_true"]), psi_true), "truth differs from the pilot's"

    res = {"psi_true": psi_true, "names": M.NAMES, "stage1_scale": scale, "datasets": []}
    for r in a.datasets:
        t0 = time.time()
        rng = np.random.default_rng([42, 45, 1, r])
        sYT, sYP = P.simulate_groups(rng, groups, psi_true, it_true)
        sg = P.regroup(groups, sYT, sYP)
        row = {"dataset": r}
        if a.check_dataset0 and r == 0:
            fT, _, _ = P.fit_items_multistart(P.flatten(sg)[0].astype(float), 100 + r, starts=1)
            fP, _, _ = P.fit_items_multistart(P.flatten(sg)[1].astype(float), 200 + r, starts=1)
            got = {"a_maxerr_T": float(np.max(np.abs(fT["a"] - it_true.aT))),
                   "a_maxerr_P": float(np.max(np.abs(fP["a"] - it_true.aP)))}
            want = {k: pilot_rec["items"][0][k] for k in got}
            row["identity_check"] = {"got": got, "pilot": want, "identical": got == want}
            if got != want:
                (out / "results.json").write_text(json.dumps(res | {"aborted": row}, indent=2, default=_json))
                print("synthetic dataset 0 differs from the pilot's", file=sys.stderr)
                return 3
        fit = M.fit(sg, it_true)                                   # stage 2 with items = truth
        H = M.hessian(fit.psi, sg, it_true)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            se = np.full(M.N_PSI, np.nan)
        z = (fit.psi - psi_true) / se
        row.update({"nll": fit.nll, "converged": fit.converged, "grad_norm": fit.grad_norm,
                    "at_bound": fit.at_bound, "hessian_min_eig": float(np.linalg.eigvalsh(H).min()),
                    "psi_hat": fit.psi, "se": se, "z": z, "covered": np.abs(z) <= 1.96,
                    "runtime_s": time.time() - t0})
        res["datasets"].append(row)
        print(f"[{time.strftime('%H:%M:%S')}] dataset {r}: covered {int(row['covered'].sum())}/{M.N_PSI}, "
              f"Le00 z {z[M.NAMES.index('Le00_log')]:+.2f}, Le11 z {z[M.NAMES.index('Le11_log')]:+.2f}, "
              f"beta0_P z {z[M.NAMES.index('beta0_P')]:+.2f}, {row['runtime_s']:.0f}s", flush=True)
    cov = np.sum([d["covered"] for d in res["datasets"]], axis=0)
    res["coverage_counts"] = cov
    res["n_datasets"] = len(res["datasets"])
    res["criterion"] = "covered in >= 4 of 5 (doc 16 D3); evaluated only when 5 datasets are run"
    res["passed"] = bool(np.all(cov >= 4)) if len(res["datasets"]) == 5 else None
    res["runtime_s"] = time.time() - t_start
    (out / "results.json").write_text(json.dumps(res, indent=2, default=_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
