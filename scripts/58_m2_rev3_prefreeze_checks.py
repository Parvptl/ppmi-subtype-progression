"""PRE-FREEZE READ-ONLY CHECKS for the Rev 3 recovery protocol (docs 46/47 R1.3; review findings B5).

Read-only on saved D4 / Step 6 outputs; the five old development datasets are regenerated in memory only
to evaluate item-objective gradients (nothing is fitted, nothing is written except this output directory).

  N6  condition number of the 198 × 198 stacked Jacobian A (blocks A1, C1, H1, E3, A3, C2, H2), 2-norm,
      unscaled (as specified) and, for information, after symmetric Jacobi scaling diag(|A_ii|)^(-1/2).
  N4  the S1-b item-stage relabelling identity, with and without the exact first-order gradient term:
        A3 v_φ + E3 v_ψ − [c-direction] m_φ(s) ⊙ g3 = 0,   relative to ‖A3 v_φ‖,
      where g3 = ∇_φ NLL3(φ̂2; ψ̂1) (the S1-b item objective gradient at the saved estimate) and m_φ(s)
      marks the first-threshold entries γ1 of subscore s (the only packed item coordinates whose Jacobian
      under θ' = (θ − d)/c depends on c).

    python scripts/58_m2_rev3_prefreeze_checks.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402

S6 = ROOT / "experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6"
DIAG = ROOT / "experiments/20261019T084559Z_stage2_step6_diagnostics"
KT, KP = 11, 5
NT, NP = 5 * KT, 5 * KP


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gamma1_mask(sub):
    m = np.zeros(NT + NP, bool)
    if sub == "T":
        m[[5 * k + 1 for k in range(KT)]] = True
    else:
        m[[NT + 5 * k + 1 for k in range(KP)]] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    D54 = load_module("d54_pf", ROOT / "scripts/54_m2_step6_diag_compute.py")
    R53 = load_module("r53_pf", ROOT / "scripts/53_m2_step6_diag_readonly.py")
    S1A = load_module("s1a_pf", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")
    P = load_module("pilot_pf", ROOT / "scripts/45_m2_pilot.py")
    from ppmi_subtype.config import load_config
    _, t, x, yt, yp, *_ = P.load_cohort(load_config())
    g0 = M.make_groups(t, x, yt, yp)
    psi_true, it_true, _ = P.truth_on_stage1_scale(g0)
    res = {"label": "PRE-FREEZE READ-ONLY CHECKS — not a recovery result", "datasets": []}
    nodes = G.gh_nodes()
    for r in range(5):
        d6 = json.loads((S6 / f"S1_dataset{r}.json").read_text())
        d4 = json.loads((DIAG / f"d4/D4_dataset{r}.json").read_text())
        A1, A3, E3, C1, C2 = (np.array(d4[k]) for k in ("A1", "A3", "E3", "C1", "C2"))
        H1, H2 = np.array(d6["H1_nonbinding"]), np.array(d6["H2"])
        p = 19
        n = 2 * 80 + 2 * p
        J = np.zeros((n, n))
        i1, i2, i3, i4 = slice(0, 80), slice(80, 99), slice(99, 179), slice(179, 198)
        J[i1, i1] = A1
        J[i2, i1], J[i2, i2] = C1, H1
        J[i3, i2], J[i3, i3] = E3, A3
        J[i4, i3], J[i4, i4] = C2, H2
        s = np.linalg.svd(J, compute_uv=False)
        dsc = 1 / np.sqrt(np.abs(np.diag(J)))
        ss = np.linalg.svd(J * dsc[:, None] * dsc[None, :], compute_uv=False)
        # N4 with the saved S1-b items φ̂2 and ψ̂1
        psi1 = np.array(d6["stage2_fit_1"]["psi"])
        it2 = M.Items(*(np.array(d6["items_2"][q][k]) for q in ("T", "P") for k in ("a", "tau")))
        phi2 = np.concatenate(D54.pack_set(it2))
        sg = P.regroup(g0, *P.simulate_groups(np.random.default_rng([42, 45, 1, r]), g0, psi_true, it_true))
        YT, YP, _ = P.flatten(sg)
        vl = S1A.visit_latent(sg, psi1)
        g3 = np.concatenate([S1A.latent_negloglik(phi2[:NT], YT.astype(float), *vl["T"], nodes)[1],
                             S1A.latent_negloglik(phi2[NT:], YP.astype(float), *vl["P"], nodes)[1]])
        n4 = {}
        for sub in ("T", "P"):
            for par in ("c", "d"):
                vphi, vpsi = D54.affine_tangent(it2, psi1, sub, par, R53.relabel_psi)
                base = A3 @ vphi + E3 @ vpsi
                corr = gamma1_mask(sub) * g3 if par == "c" else np.zeros_like(g3)
                den = np.linalg.norm(A3 @ vphi)
                n4[f"{sub}_{par}"] = {"rel_without_gradient_term": float(np.linalg.norm(base) / den),
                                      "rel_with_gradient_term": float(np.linalg.norm(base - corr) / den)}
        res["datasets"].append({"dataset": r, "cond2_unscaled": float(s[0] / s[-1]), "smax": float(s[0]),
                                "smin": float(s[-1]), "cond2_jacobi_scaled": float(ss[0] / ss[-1]),
                                "g3_norm": float(np.linalg.norm(g3)), "N4": n4})
        print(r, f"cond {s[0] / s[-1]:.3e} scaled {ss[0] / ss[-1]:.3e}",
              {k: (f"{v['rel_without_gradient_term']:.1e}", f"{v['rel_with_gradient_term']:.1e}") for k, v in n4.items()})
    res["inputs_sha256"] = {str(q.relative_to(ROOT)): sha(q) for q in [S6 / f"S1_dataset{r}.json" for r in range(5)]
                            + [DIAG / f"d4/D4_dataset{r}.json" for r in range(5)]}
    res["script_sha256"] = sha(Path(__file__))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    (out / "MANIFEST.sha256").write_text(f"{sha(out / 'summary.json')}  summary.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
