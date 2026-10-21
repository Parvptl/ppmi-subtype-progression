"""PLANNING VERIFICATION (thresholds) for the proposed Rev 3 recovery test. Not a recovery test.

Reads the gate correlation saved by script 56 (mean over the five development datasets of the full-chain
covariance propagated to the 19 re-expressed parameters and the 17 functionals) and evaluates, by Monte
Carlo of standard-normal z-vectors only (seed [42, 50, 1]), the decision rule

    coverage >= COV_MIN of R   and   |mean z| <= THR      for every gate,

for THR in {0.30, 0.40, 0.50} and COV_MIN = 88, R = 100, under calibrated, under-coverage and bias scenarios.
No model is fitted and no synthetic PPMI data are generated.

    python scripts/57_m2_rev3_planning_thresholds.py --plan <script-56 output dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

R, COV_MIN, NTRIAL = 100, 88, 20000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    d = Path(a.plan)
    Rm = np.load(d / "gate_correlation_mean.npy")
    s56 = json.loads((d / "summary.json").read_text())
    gn = list(s56["dev_mean_z"].keys())
    ev, evec = np.linalg.eigh(Rm)
    Lf = evec * np.sqrt(np.clip(ev, 0, None))
    iLe00 = gn.index("Le00_log")
    n = len(gn)
    scen = {"calibrated": (np.zeros(n), 1.0), "se_under_10pct": (np.zeros(n), 1.1), "se_under_20pct": (np.zeros(n), 1.2),
            "all_bias_0.10": (np.full(n, 0.10), 1.0), "all_bias_0.15": (np.full(n, 0.15), 1.0),
            "all_bias_0.20": (np.full(n, 0.20), 1.0), "Le00_bias_0.3": (np.eye(n)[iLe00] * 0.3, 1.0),
            "Le00_bias_0.5": (np.eye(n)[iLe00] * 0.5, 1.0), "Le00_bias_0.7": (np.eye(n)[iLe00] * 0.7, 1.0),
            "Le00_bias_1.0": (np.eye(n)[iLe00] * 1.0, 1.0)}
    rng = np.random.default_rng([42, 50, 1])
    out = {"seed": [42, 50, 1], "trials": NTRIAL, "R": R, "cov_min": COV_MIN, "results": {}}
    for nm, (mu, k) in scen.items():
        cnt = {t: 0 for t in (0.30, 0.40, 0.50)}
        cov_all = 0
        for _ in range(NTRIAL // 500):
            z = k * (rng.standard_normal((500, R, n)) @ Lf.T) + mu
            cov = (np.abs(z) <= 1.96).sum(axis=1) >= COV_MIN
            mz = np.abs(z.mean(axis=1))
            cov_all += np.sum(cov.all(1))
            for t in cnt:
                cnt[t] += np.sum((cov & (mz <= t)).all(1))
        out["results"][nm] = {"P_pass_all_36_cov_only": cov_all / NTRIAL,
                              **{f"P_pass_all_36_thr_{t:.2f}": c / NTRIAL for t, c in cnt.items()}}
    p = d / "thresholds.json"
    p.write_text(json.dumps(out, indent=1))
    m = d / "MANIFEST.sha256"
    m.write_text(m.read_text() + f"{hashlib.sha256(p.read_bytes()).hexdigest()}  thresholds.json\n")
    for nm, v in out["results"].items():
        print(nm, {k: round(x, 4) for k, x in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
