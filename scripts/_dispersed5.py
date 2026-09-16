"""Dispersed-start probe for the 5-hidden-state HMM.

The project's restart procedure perturbs a single fixed starting point by
N(0, 0.4). For the five-state model every outer seed returned a bit-identical
solution, which means the unperturbed start always won and the perturbations
never escaped it. That says nothing about whether the likelihood surface has
other optima.

This probe starts the rate block from widely separated values instead, using
the `init_rates` argument the fitter already exposes. It is a diagnostic, not a
change to the fitting protocol: the leave-one-out run still uses the project's
own restart procedure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import hmm  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

cfg = load_config()
L = cfg.subtype.labels
states = [L["TD"], L["Indeterminate"], L["PIGD"]]
names = ["TD", "Indeterminate", "PIGD"]

panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
panel = panel[panel["subtype"].notna()].sort_values(["PATNO", "years_from_baseline"])
seqs = hmm.build_sequences(panel, states)

n_rates = 5 * 4
rng = np.random.default_rng(0)
starts = {
    "project default (0.15)": np.full(n_rates, 0.15),
    "slow (0.01)": np.full(n_rates, 0.01),
    "fast (1.0)": np.full(n_rates, 1.0),
    "random A": rng.uniform(0.01, 1.0, n_rates),
    "random B": rng.uniform(0.01, 1.0, n_rates),
    "random C": rng.uniform(0.01, 1.0, n_rates),
}

out = []
for label, init in starts.items():
    t = time.time()
    f = hmm.fit_hmm(seqs, states, state_names=names, seed=cfg.seed,
                    n_restarts=cfg.model["n_restarts"], init_rates=init,
                    emission_floor=None, n_hidden=5)
    dt = time.time() - t
    out.append({"start": label, "loglik": f.loglik, "converged": bool(f.converged),
                "secs": dt, "E": f.E.tolist(), "Q": f.Q.tolist()})
    print(f"{label:<24} loglik {f.loglik:12.4f}  conv {str(f.converged):5s}  {dt:5.0f}s",
          flush=True)

dest = cfg.root / "experiments" / "hidden_5_observed_3"
dest.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dest / "dispersed_start_fits.json", "w"), indent=1)

best = max(x["loglik"] for x in out)
worst = min(x["loglik"] for x in out)
print(f"\nspread across starts: {best - worst:.4f} log-likelihood units")
print("wrote", dest / "dispersed_start_fits.json")
