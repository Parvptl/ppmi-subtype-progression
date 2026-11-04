"""What the ratio bands tell the latent state: a descriptive whole-cohort fit.

The LOPO driver (`scripts/87_...`) saves predictions, not parameters: each of
its 934 folds fits its own model and none of them is *the* model. To describe
the mechanism --- whether a hidden state's distribution over ratio bands
differs from its neighbours' at all --- one fit on the whole cohort is needed.

This fit is DESCRIPTIVE ONLY. It sees every patient, so nothing here is a
held-out result and no number from this script is evidence about predictive
performance. It answers one question: given the alphabet, did the emission
actually use it?

    python scripts/88_ratio_band_emission_diagnostic.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hmm_banded, hmm_stability, loo  # noqa: E402
from ppmi_subtype import next_visit as nv, next_visit_banded as nvb  # noqa: E402
from ppmi_subtype import ratio_bands as rb  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def block(frame, **kw) -> str:
    return "    " + frame.to_string(**kw).replace("\n", "\n    ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = hashlib.md5(panel_path.read_bytes()).hexdigest()
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != {EXPECTED_PANEL_MD5}")

    panel = pd.read_csv(panel_path, low_memory=False)
    states = [cfg.subtype.labels[s] for s in nv.STATE_NAMES]
    data = loo.prepare(panel, states)
    bdata = nvb.prepare(panel, data, cfg.subtype)
    scheme, alphabet, _ = nvb.fold_alphabet(bdata, None)

    say("DESCRIPTIVE WHOLE-COHORT FIT -- NOT A HELD-OUT RESULT")
    say("=" * 70)
    say(f"  panel md5            {panel_md5}  [MATCH]")
    say(f"  patients             {len(data.sequences):,}")
    say(f"  labelled visits      {sum(len(s['labels']) for s in bdata.sequences):,}")
    say(f"  joint categories     {alphabet.n_categories}")
    say(f"  seed                 {cfg.seed}   restarts {cfg.model['n_restarts']}")
    say("")
    say("  Every patient is in this fit. It describes the fitted emission; it")
    say("  says nothing about prediction, which scripts/87 measures out of sample.")

    started = time.time()
    joint = nvb.joint_sequences(bdata, alphabet)
    banded, records = hmm_banded.fit_banded(
        joint, alphabet, nv.STATE_NAMES, emission_floor=cfg.model["emission_floor"],
        seed=cfg.seed, n_restarts=cfg.model["n_restarts"])
    starts = nv.audited_starts(3, 3, cfg.model["emission_floor"], cfg.seed,
                               cfg.model["n_restarts"])
    label_fit, label_records = hmm_stability.fit_multistart(
        data.sequences, states, n_hidden=3,
        emission_floor=cfg.model["emission_floor"], starts=starts,
        state_names=nv.STATE_NAMES)

    say("")
    say(f"  banded   loglik {banded.loglik:12.4f}  params {banded.n_params:>4}  "
        f"AIC {banded.aic:12.2f}")
    say(f"  baseline loglik {label_fit.loglik:12.4f}  params {label_fit.n_params:>4}  "
        f"AIC {label_fit.aic:12.2f}")
    say("  The two log-likelihoods are over DIFFERENT observation alphabets (44")
    say("  categories against 3), so they are not comparable to each other and")
    say("  neither is their AIC. Both are reported only as fit diagnostics.")
    say(f"  fitted in {time.time() - started:.0f}s; restart logliks: "
        f"{[round(r.loglik, 3) for r in records]}")

    say("")
    say("  Subtype misclassification M of the joint model (rows latent state):")
    say(block(banded.subtype_emission_frame().round(4)))
    say("")
    say("  Emission E of the baseline model, the same quantity:")
    say(block(label_fit.emission_frame().round(4)))
    say("")
    say("  Generator Q, joint model (per year):")
    say(block(banded.generator_frame().round(4)))
    say("")
    say("  Generator Q, baseline model:")
    say(block(label_fit.generator_frame().round(4)))

    banded.subtype_emission_frame().round(6).to_csv(out / "emission_subtype_banded.csv")
    label_fit.emission_frame().round(6).to_csv(out / "emission_baseline.csv")
    banded.generator_frame().round(6).to_csv(out / "generator_banded.csv")
    label_fit.generator_frame().round(6).to_csv(out / "generator_baseline.csv")

    cond = banded.band_conditional_frame()
    cond.round(6).to_csv(out / "band_conditionals.csv", index=False)

    # Does the band distribution differ across hidden states? If P(band |
    # state, subtype) were the same for every state the band would carry no
    # information about the latent state and the model would be the baseline
    # with extra parameters.
    say("")
    say("  Does the band distribution depend on the latent state?")
    say("  For each subtype, the largest total-variation distance between two")
    say("  latent states' distributions over that subtype's bands. Zero would")
    say("  mean the bands say nothing about the latent state.")
    rows = []
    for s, name in enumerate(nv.STATE_NAMES):
        blockcols = list(alphabet.blocks[s])
        C = banded.E[:, blockcols] / banded.M[:, s][:, None]
        tv = max(0.5 * np.abs(C[a] - C[b]).sum()
                 for a in range(3) for b in range(3) if a < b)
        ratios = np.concatenate([seq["ratios"] for seq in bdata.sequences])
        labels = np.concatenate([seq["labels"] for seq in bdata.sequences])
        obs = ratios[(labels == s) & np.isfinite(ratios)]
        rows.append({
            "subtype": name,
            "n_bands": len(blockcols),
            "max_total_variation_between_states": tv,
            "observed_ratio_iqr": (float(np.percentile(obs, 75)
                                         - np.percentile(obs, 25))
                                   if len(obs) else float("nan")),
            "observed_ratio_median": float(np.median(obs)) if len(obs) else float("nan"),
        })
    spread = pd.DataFrame(rows)
    spread.round(4).to_csv(out / "band_information.csv", index=False)
    say(block(spread.round(4), index=False))

    # The most informative individual categories: where the states disagree most.
    say("")
    say("  The ten joint categories whose emission probability differs most")
    say("  between latent states (ratio of largest to smallest, mass-weighted):")
    pivot = cond.pivot(index="category", columns="hidden_state", values="p_joint")
    pivot = pivot[[n for n in nv.STATE_NAMES if n in pivot.columns]]
    occ = nvb.band_occupancy(bdata, alphabet).set_index("category")["n_observations"]
    pivot["n_obs"] = occ.reindex(pivot.index).fillna(0).astype(int)
    pivot["max_over_min"] = (pivot[nv.STATE_NAMES].max(axis=1)
                             / pivot[nv.STATE_NAMES].min(axis=1).replace(0, np.nan))
    top = pivot[pivot["n_obs"] >= 15].sort_values("max_over_min", ascending=False).head(10)
    say(block(top.round(5)))
    pivot.round(6).to_csv(out / "category_emission_by_state.csv")

    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "descriptive_only": True,
        "panel_md5": panel_md5,
        "n_patients": len(data.sequences),
        "n_categories": alphabet.n_categories,
        "banded_loglik": banded.loglik, "banded_params": banded.n_params,
        "baseline_loglik": label_fit.loglik, "baseline_params": label_fit.n_params,
        "band_information": spread.to_dict(orient="records"),
        "restart_logliks_banded": [r.loglik for r in records],
        "restart_logliks_baseline": [r.loglik for r in label_records],
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
