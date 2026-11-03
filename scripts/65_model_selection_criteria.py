"""AIC/BIC for the two methods, and their held-out TD/PIGD metrics on matched cases.

Neither model is modified. Each is fitted once on the full cohort purely to
obtain a likelihood at its own maximum, which is what AIC and BIC need; all
held-out performance figures are read from the existing experiment directories
rather than recomputed.

The central point is that **AIC and BIC are not comparable between the two
methods**, and the script groups the models accordingly:

  Group A  rich-profile variants -- identical Gaussian density over the same
           nine continuous domains, and nested, so AIC/BIC are valid.
  Group B  observed-label chains (CTMC, discrete Markov) -- identical
           conditional likelihood over the same transition pairs, so AIC/BIC
           are valid between them.
  Alone    the CT-HMM -- its likelihood is a *joint* over whole label
           sequences including the initial state, where Group B's is
           *conditional* on the previous observed state. Its AIC is reported
           but is not on the same scale as Group B's.

Across groups nothing is comparable: Group A is a probability *density* over
continuous measurements, Group B and the CT-HMM are probability *masses* over
categorical labels, on a different number of observations.

    python scripts/65_model_selection_criteria.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import clinical_profile as cp  # noqa: E402
from ppmi_subtype import clinical_states as cs  # noqa: E402
from ppmi_subtype import ctmc, hmm, loo  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
RP_RUN = "experiments/20261020T180515Z_clinical_profile_cv"

lines: list[str] = []


def say(t: str = "") -> None:
    print(t, flush=True)
    lines.append(t)


def section(t: str) -> None:
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def block(f, **kw) -> str:
    return "    " + f.to_string(**kw).replace("\n", "\n    ")


def criteria(loglik: float, k: int, n: int) -> tuple[float, float]:
    """AIC and BIC. `n` is the number of terms in the likelihood."""
    return 2 * k - 2 * loglik, k * np.log(max(n, 2)) - 2 * loglik


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    section("1. INPUT")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    pmd5 = md5(panel_path)
    if pmd5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {pmd5} != expected")
    say(f"  panel md5 {pmd5}  [MATCH]")
    panel = pd.read_csv(panel_path, low_memory=False)

    states = [cfg.subtype.labels[s] for s in loo.STATE_NAMES]
    data = loo.prepare(panel, states)
    clinical = cs.build_clinical_panel(cfg, panel)
    rp_seqs = cp.build_sequences(clinical)

    say(f"  CT-HMM cohort      {len(data.sequences):,} patients, "
        f"{data.n_transitions:,} transitions")
    say(f"  rich-profile cohort {len(rp_seqs):,} patients, "
        f"{sum(len(s['gaps']) for s in rp_seqs):,} transitions")
    say("")
    say("  Each model is fitted ONCE on the full cohort below, only to obtain a")
    say("  likelihood at its own maximum. Neither model is changed, and every")
    say("  held-out number in section 4 comes from the existing runs.")

    rows = []

    # ---------------------------------------------------- categorical models
    section("2. MODELS OF THE CATEGORICAL SUBTYPE LABEL")
    say("  Likelihood is a probability MASS over TD / Indeterminate / PIGD.")

    t = time.time()
    h = hmm.fit_hmm(data.sequences, states, state_names=loo.STATE_NAMES,
                    seed=cfg.seed, n_restarts=cfg.model["n_restarts"],
                    emission_floor=cfg.model["emission_floor"])
    n_obs_hmm = h.n_observations
    a, b = criteria(h.loglik, h.n_params, n_obs_hmm)
    say(f"\n  CT-HMM fitted in {time.time()-t:.1f}s")
    say(f"    likelihood object : JOINT over whole sequences, incl. initial state")
    say(f"    log-likelihood    : {h.loglik:,.2f}")
    say(f"    parameters        : {h.n_params}   "
        f"(6 rates + 6 emission + 2 initial)")
    say(f"    likelihood terms  : {n_obs_hmm:,} visits")
    say(f"    AIC {a:,.2f}    BIC {b:,.2f}")
    rows.append({"group": "CT-HMM (alone)", "model": "CT-HMM (3 latent states)",
                 "likelihood": "joint PMF over label sequences",
                 "loglik": h.loglik, "k": h.n_params, "n_terms": n_obs_hmm,
                 "aic": a, "bic": b})

    say("\n  Observed-label chains (no latent layer). Their likelihood is")
    say("  CONDITIONAL on the previous observed state, over transition pairs --")
    say("  a different object from the CT-HMM's joint likelihood.")
    c = ctmc.fit_ctmc(data.pairs, states, state_names=loo.STATE_NAMES,
                      seed=cfg.seed, n_restarts=cfg.model["n_restarts"])
    a, b = criteria(c.loglik, c.n_params, c.n_pairs)
    say(f"\n    CTMC   loglik {c.loglik:,.2f}  k {c.n_params}  "
        f"n {c.n_pairs:,}   AIC {a:,.2f}  BIC {b:,.2f}")
    rows.append({"group": "B: observed-label chains", "model": "CTMC (interval-aware)",
                 "likelihood": "conditional PMF over pairs",
                 "loglik": c.loglik, "k": c.n_params, "n_terms": c.n_pairs,
                 "aic": a, "bic": b})

    # Discrete chain: the same conditional likelihood, evaluated at its own MLE.
    dt = ctmc.fit_dtmc(data.pairs, states)
    idx = {s: k for k, s in enumerate(states)}
    i = data.pairs["from_state"].map(idx).to_numpy(int)
    j = data.pairs["to_state"].map(idx).to_numpy(int)
    ll_d = float(np.log(np.maximum(dt[i, j], 1e-300)).sum())
    k_d = len(states) * (len(states) - 1)
    a, b = criteria(ll_d, k_d, len(data.pairs))
    say(f"    Markov loglik {ll_d:,.2f}  k {k_d}  n {len(data.pairs):,}   "
        f"AIC {a:,.2f}  BIC {b:,.2f}")
    rows.append({"group": "B: observed-label chains", "model": "Markov (discrete)",
                 "likelihood": "conditional PMF over pairs",
                 "loglik": ll_d, "k": k_d, "n_terms": len(data.pairs),
                 "aic": a, "bic": b})

    # -- making the latent layer comparable to the chains ------------------
    say("\n  The CT-HMM's joint likelihood covers exactly the transition pairs")
    say(f"  PLUS each patient's first visit: {n_obs_hmm:,} - {len(data.pairs):,} = "
        f"{n_obs_hmm - len(data.pairs):,} = the number of patients.")
    say("  So the chains can be put on the CT-HMM's scale by giving them an")
    say("  initial-state distribution, estimated from the first visits (2 free")
    say("  parameters). That makes the latent layer's cost directly testable.")

    first = np.array([s["labels"][0] for s in data.sequences])
    pi_hat = np.bincount(first, minlength=len(states)) / len(first)
    ll_init = float(np.log(np.maximum(pi_hat[first], 1e-300)).sum())
    say(f"\n    initial-state distribution  {np.round(pi_hat, 4).tolist()}")
    say(f"    initial-state log-likelihood {ll_init:,.2f}  (+2 parameters)")

    say("")
    for nm, ll_pairs, kk in (("CTMC + initial state", c.loglik, c.n_params + 2),
                             ("Markov + initial state", ll_d, k_d + 2)):
        joint = ll_pairs + ll_init
        a2, b2 = criteria(joint, kk, n_obs_hmm)
        say(f"    {nm:<24} loglik {joint:,.2f}  k {kk}  "
            f"AIC {a2:,.2f}  BIC {b2:,.2f}")
        rows.append({"group": "C: joint likelihood over whole sequences",
                     "model": nm, "likelihood": "joint PMF over label sequences",
                     "loglik": joint, "k": kk, "n_terms": n_obs_hmm,
                     "aic": a2, "bic": b2})
    rows.append({"group": "C: joint likelihood over whole sequences",
                 "model": "CT-HMM (3 latent states)",
                 "likelihood": "joint PMF over label sequences",
                 "loglik": h.loglik, "k": h.n_params, "n_terms": n_obs_hmm,
                 "aic": 2 * h.n_params - 2 * h.loglik,
                 "bic": h.n_params * np.log(n_obs_hmm) - 2 * h.loglik})

    # ------------------------------------------------------- profile models
    section("3. MODELS OF THE NINE CONTINUOUS CLINICAL DOMAINS")
    say("  Likelihood is a probability DENSITY. These three are nested and")
    say("  share one likelihood object, so AIC/BIC are valid between them.")
    std = cp.Standardiser.fit(rp_seqs)
    z = std.apply(rp_seqs)
    n_terms_rp = int(np.isfinite(np.vstack([s["Y"] for s in z])).sum())

    for name, kw in ((cp.MODEL_FULL, {"use_rate": True, "use_medication": True}),
                     (cp.MODEL_NO_RATE, {"use_rate": False, "use_medication": True}),
                     (cp.MODEL_NO_MED, {"use_rate": True, "use_medication": False})):
        t = time.time()
        m = cp.fit(z, **kw)
        a, b = criteria(m.loglik, m.n_params, n_terms_rp)
        say(f"\n    {name}")
        say(f"      loglik {m.loglik:,.2f}  k {m.n_params}  "
            f"n {n_terms_rp:,}  ({time.time()-t:.0f}s, converged={m.converged})")
        say(f"      AIC {a:,.2f}   BIC {b:,.2f}")
        rows.append({"group": "A: rich-profile variants", "model": name,
                     "likelihood": "density over 9 continuous domains",
                     "loglik": m.loglik, "k": m.n_params,
                     "n_terms": n_terms_rp, "aic": a, "bic": b})

    table = pd.DataFrame(rows)
    for g, sub in table.groupby("group", sort=False):
        table.loc[sub.index, "delta_aic_within_group"] = sub["aic"] - sub["aic"].min()
        table.loc[sub.index, "delta_bic_within_group"] = sub["bic"] - sub["bic"].min()
    table.round(2).to_csv(out / "aic_bic.csv", index=False)

    say("\n\n  Within-group comparison (lower is better; Δ is to the group's best):")
    say(block(table[["group", "model", "loglik", "k", "n_terms", "aic", "bic",
                     "delta_aic_within_group", "delta_bic_within_group"]].round(2),
              index=False))

    # ------------------------------------------------------- held-out, matched
    section("4. HELD-OUT TD/PIGD PREDICTION ON MATCHED CASES (existing outputs)")
    paired = pd.read_csv(ROOT / RP_RUN / "derived_label_vs_subtype_history.csv")
    contrasts = pd.read_csv(ROOT / RP_RUN / "derived_label_contrasts.csv")
    show = ["model", "n_scored", "accuracy", "accuracy_ci_patient_bootstrap_low",
            "accuracy_ci_patient_bootstrap_high", "brier_two_class", "log_loss"]
    say(block(paired[show].round(4), index=False))
    paired.to_csv(out / "heldout_matched.csv", index=False)
    say("")
    say("  Paired differences (patients resampled):")
    say(block(contrasts[["contrast", "accuracy_delta",
                         "delta_ci_patient_bootstrap_low",
                         "delta_ci_patient_bootstrap_high"]].round(4), index=False))
    contrasts.to_csv(out / "heldout_contrasts.csv", index=False)

    # ---------------------------------------------------------- provenance
    section("5. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(), "seed": cfg.seed,
           "panel_md5": pmd5, "ct_hmm_run": CT_RUN, "profile_run": RP_RUN,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<18} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env,
         "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                         [Path(__file__), ROOT / "src/ppmi_subtype/hmm.py",
                          ROOT / "src/ppmi_subtype/ctmc.py",
                          ROOT / "src/ppmi_subtype/clinical_profile.py"]},
         "criteria": table.to_dict(orient="records"),
         "heldout_matched": paired.to_dict(orient="records"),
         "heldout_contrasts": contrasts.to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
