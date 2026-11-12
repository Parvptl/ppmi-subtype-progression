"""Reproduce the HMM fits the manuscript reports, and the Markov-property test.

    python scripts/_verify_hmm_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype import ctmc, hmm, splitting, transitions as T  # noqa: E402

STATES = [1.0, 2.0, 3.0]
NAMES = ["TD", "Indeterminate", "PIGD"]


def say(label: str, value) -> None:
    print(f"{label:<58} {value}")


def main() -> int:
    cfg = load_config()
    panel = pd.read_csv(ROOT / "outputs" / "visit_panel.csv", low_memory=False)

    # ---------------------------------------------------------------- Markov
    print("=" * 86)
    print("MARKOV PROPERTY IN THE OBSERVED LABEL (labelled triples)")
    print("=" * 86)
    df = panel[panel["subtype"].notna()].sort_values(["PATNO", "years_from_baseline"])
    g = df.groupby("PATNO", sort=False)
    trip = pd.DataFrame({"s0": df["subtype"], "s1": g["subtype"].shift(-1),
                         "s2": g["subtype"].shift(-2)}).dropna()
    say("labelled triples", f"{len(trip):,}")
    for k, s1 in enumerate(STATES):
        sub = trip[trip["s1"] == s1]
        tab = pd.crosstab(sub["s0"], sub["s2"])
        c2, p, dof, _ = chi2_contingency(tab)
        say(f"  currently {NAMES[k]}: n, chi2, dof, p",
            f"{len(sub):,}, {c2:.4f}, {dof}, {p:.4e}")
    sub = trip[trip["s1"] == 1.0]
    tab = pd.crosstab(sub["s0"], sub["s2"], normalize="index")
    tab.index = [NAMES[int(i) - 1] for i in tab.index]
    tab.columns = [NAMES[int(c) - 1] for c in tab.columns]
    print("\n  currently TD, by the state they arrived from (row-normalised):")
    print("   ", tab.round(4).to_string().replace("\n", "\n    "))
    say("  P(PIGD next | TD now, TD before)", f"{tab.loc['TD', 'PIGD']:.4f}")
    say("  P(PIGD next | TD now, PIGD before)", f"{tab.loc['PIGD', 'PIGD']:.4f}")
    counts_tab = pd.crosstab(sub["s0"], sub["s2"])
    print("\n  counts:\n   ", counts_tab.to_string().replace("\n", "\n    "))

    # --------------------------------------------------- full-cohort HMM fit
    print()
    print("=" * 86)
    print("HMM FITTED ON ALL PATIENTS WITH >= 2 LABELLED VISITS")
    print("=" * 86)
    seq_all = hmm.build_sequences(panel, STATES)
    say("sequences", f"{len(seq_all):,}")
    say("observations", f"{sum(len(s['labels']) for s in seq_all):,}")
    say("transitions", f"{sum(len(s['gaps']) for s in seq_all):,}")

    fit = hmm.fit_hmm(seq_all, STATES, NAMES, seed=cfg.seed, n_restarts=4,
                      emission_floor=cfg.model["emission_floor"])
    say("converged / loglik / n_params",
        f"{fit.converged} / {fit.loglik:.4f} / {fit.n_params}")
    say("AIC", f"{fit.aic:.4f}")
    print("\nlatent generator Q (per year):")
    print("   ", fit.generator_frame().round(4).to_string().replace("\n", "\n    "))
    print("\nemission matrix E  (rows = latent state, cols = recorded label):")
    print("   ", fit.emission_frame().round(4).to_string().replace("\n", "\n    "))
    print("\nmean latent sojourn (years):")
    print("   ", fit.mean_sojourn().round(3).to_string().replace("\n", "\n    "))
    print("\ninitial latent distribution:")
    print("   ", pd.Series(fit.pi, index=NAMES).round(4).to_string().replace("\n", "\n    "))

    # observed-label persistence for comparison
    pairs_all = T.consecutive_pairs(panel)
    dt = ctmc.fit_ctmc(pairs_all, STATES, NAMES, seed=cfg.seed, n_restarts=4)
    print("\nobserved-label CTMC generator (same data, no latent layer):")
    print("   ", pd.DataFrame(dt.Q, index=NAMES, columns=NAMES).round(4)
          .to_string().replace("\n", "\n    "))
    print("\nobserved-label mean sojourn (years):")
    d = np.diag(dt.Q)
    print("   ", pd.Series(np.where(d < 0, -1 / d, np.inf), index=NAMES).round(3)
          .to_string().replace("\n", "\n    "))

    # latent occupancy at clinical horizons
    print("\nlatent-state occupancy from the full-cohort fit:")
    print(hmm.clinical_table(fit, horizons=(1, 2, 3, 5)).round(4).to_string(index=False))

    # ----------------------------------------- emission-floor sensitivity
    print()
    print("=" * 86)
    print("EMISSION-FLOOR SENSITIVITY OF THE FULL-COHORT FIT")
    print("=" * 86)
    rows = []
    for floor in [None, 0.5, 0.6, 0.7]:
        f = hmm.fit_hmm(seq_all, STATES, NAMES, seed=cfg.seed, n_restarts=4,
                        emission_floor=floor)
        rows.append({"floor": "free" if floor is None else f"{floor:.2f}",
                     "loglik": f.loglik, "aic": f.aic,
                     "E_TD_TD": f.E[0, 0], "E_Ind_Ind": f.E[1, 1],
                     "E_PIGD_PIGD": f.E[2, 2],
                     "sojourn_TD": f.mean_sojourn().iloc[0],
                     "sojourn_Ind": f.mean_sojourn().iloc[1],
                     "sojourn_PIGD": f.mean_sojourn().iloc[2]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ----------------------------------------- number of latent states
    print()
    print("=" * 86)
    print("LATENT-STATE COUNT (unconstrained emissions, full cohort)")
    print("=" * 86)
    rows = []
    for nh in [3, 4, 5]:
        f = hmm.fit_hmm(seq_all, STATES, NAMES, seed=cfg.seed, n_restarts=4,
                        emission_floor=None, n_hidden=nh)
        n_obs = sum(len(s["labels"]) for s in seq_all)
        rows.append({"n_hidden": nh, "n_params": f.n_params, "loglik": f.loglik,
                     "AIC": f.aic,
                     "BIC": f.n_params * np.log(n_obs) - 2 * f.loglik,
                     "converged": f.converged})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
