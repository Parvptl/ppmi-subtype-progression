"""Multi-start stability probe for the 5-hidden-state HMM.

Not part of the pipeline; a diagnostic run to answer whether different
initialisations land on the same solution before committing to leave-one-out.
Fits the full dataset, so nothing here is held out and no result from it should
be read as predictive performance.
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
print(f"{len(seqs)} sequences", flush=True)

out = []
for seed in (42, 1, 7, 123, 2024, 99):
    t = time.time()
    f = hmm.fit_hmm(seqs, states, state_names=names, seed=seed,
                    n_restarts=cfg.model["n_restarts"],
                    emission_floor=None, n_hidden=5)
    dt = time.time() - t
    out.append({
        "seed": seed, "loglik": f.loglik, "converged": bool(f.converged),
        "secs": dt, "E": f.E.tolist(), "Q": f.Q.tolist(), "pi": f.pi.tolist(),
    })
    print(f"seed {seed:5d}  loglik {f.loglik:12.4f}  conv {str(f.converged):5s}  {dt:5.0f}s",
          flush=True)

dest = cfg.root / "experiments" / "hidden_5_observed_3"
dest.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dest / "multistart_fits.json", "w"), indent=1)
print("wrote", dest / "multistart_fits.json")
