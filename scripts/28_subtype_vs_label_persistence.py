"""Paired subtype comparison against LABEL persistence (repeat the last label).

Script 22 compares derived subtypes with the HMM, and with `persistence` only
through that arm's Gaussian error model. The clinically meaningful baseline is
simpler: predict that the next recorded label equals the current one. On the
subscore scale that is exactly the `point` rule applied to the persistence arm
(its predictive mean is the last visit's subscores, and the label rule is
deterministic). This script pairs every model's derived subtype against it,
for overall accuracy, balanced accuracy and Indeterminate recall, with the same
shared patient bootstrap (4,000 replicates, seed 42) as script 22.

Reads only <run_dir>/analysis/subtype_predictions.csv; writes
<run_dir>/analysis/subtype_paired_vs_label_persistence.csv and nothing else.

    python scripts/28_subtype_vs_label_persistence.py <stage3_run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppmi_subtype import heldout as H  # noqa: E402

N_BOOT, SEED = 4000, 42


def balanced_draws(boot, pat, actual, pred):
    pos = np.searchsorted(boot.patients, pat)
    rec = []
    for c in range(3):
        hit = np.bincount(pos, ((actual == c) & (pred == c)).astype(float), len(boot.patients))
        sup = np.bincount(pos, (actual == c).astype(float), len(boot.patients))
        with np.errstate(invalid="ignore", divide="ignore"):
            rec.append((boot.weights @ hit) / (boot.weights @ sup))
    return np.nanmean(np.stack(rec), axis=0)


def balanced(actual, pred):
    return float(np.mean([((actual == c) & (pred == c)).sum() / (actual == c).sum()
                          for c in range(3)]))


def main(run_dir: str) -> int:
    an = Path(run_dir) / "analysis"
    d = pd.read_csv(an / "subtype_predictions.csv")
    key = ["held_out_patient", "target_index"]
    ref = d[d.model == "persistence"].sort_values(key).reset_index(drop=True)
    act, pat = ref.actual.to_numpy(int), ref.held_out_patient.to_numpy()
    lab = ref.pred_point.to_numpy(int)                     # = the last observed label
    boot = H.SharedBootstrap.create(pat, N_BOOT, SEED)
    ba_ref = balanced_draws(boot, pat, act, lab)
    rows = []
    for model, g in d.groupby("model"):
        g = g.sort_values(key).reset_index(drop=True)
        if not (np.array_equal(g.held_out_patient, ref.held_out_patient)
                and np.array_equal(g.target_index, ref.target_index)):
            raise RuntimeError(f"{model} is not aligned with persistence")
        for rule in ("pred_point", "pred_prob_argmax", "pred_prob_prior_corrected"):
            if rule not in g or g[rule].isna().all():
                continue
            pred = g[rule].to_numpy(int)
            e, lo, hi = boot.mean_ci((pred == act).astype(float) - (lab == act).astype(float), pat)
            bd = balanced_draws(boot, pat, act, pred) - ba_ref
            rows.append({
                "model": model, "rule": rule.replace("pred_", ""),
                "accuracy": float((pred == act).mean()),
                "label_persistence_accuracy": float((lab == act).mean()),
                "accuracy_diff": e, "accuracy_diff_lo": lo, "accuracy_diff_hi": hi,
                "balanced_accuracy": balanced(act, pred),
                "label_persistence_balanced_accuracy": balanced(act, lab),
                "balanced_diff": balanced(act, pred) - balanced(act, lab),
                "balanced_diff_lo": float(np.nanpercentile(bd, 2.5)),
                "balanced_diff_hi": float(np.nanpercentile(bd, 97.5)),
                "recall_Indeterminate": float(((pred == 1) & (act == 1)).sum() / (act == 1).sum()),
                "n": len(act)})
    out = pd.DataFrame(rows)
    out.to_csv(an / "subtype_paired_vs_label_persistence.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
