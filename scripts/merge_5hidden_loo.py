"""Merge the Step 2 leave-one-out shards and check that nothing is missing.

Every check has to pass before anything is written. A shard that failed
half-way through, a patient that landed in two shards, or a fold that trained
on the wrong number of patients would all quietly corrupt the pooled result, so
they are treated as errors rather than warnings.

    python scripts/merge_5hidden_loo.py [--shards-dir DIR] [--expect-patients N]

Writes outputs/loo_5hidden/loo_predictions.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import loo  # noqa: E402
from ppmi_subtype.config import load_config, project_root  # noqa: E402

EXPECT_PATIENTS = 934
EXPECT_PREDICTIONS = 3_361
EXPECT_TRAIN_PATIENTS = 933


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--expect-patients", type=int, default=EXPECT_PATIENTS)
    ap.add_argument("--expect-predictions", type=int, default=EXPECT_PREDICTIONS)
    args = ap.parse_args()

    cfg = load_config(project_root() / "config" / "config.yaml")
    base = cfg.path("outputs") / "loo_5hidden"
    shards_dir = Path(args.shards_dir) if args.shards_dir else base / "shards"

    files = sorted(shards_dir.glob("shard_*.csv"))
    if not files:
        raise SystemExit(f"no shard files found in {shards_dir}")
    print(f"found {len(files)} shard file(s) in {shards_dir}\n")

    frames, per_shard = [], []
    for f in files:
        d = pd.read_csv(f)
        hmm_rows = d[d["model"] == "HMM"]
        per_shard.append({
            "shard": f.stem,
            "rows": len(d),
            "patients": hmm_rows["held_out_patient"].nunique(),
            "hmm_rows": len(hmm_rows),
            "models": d["model"].nunique(),
        })
        frames.append(d)

    print(pd.DataFrame(per_shard).to_string(index=False))
    allrows = pd.concat(frames, ignore_index=True)
    hmm = allrows[allrows["model"] == "HMM"].reset_index(drop=True)

    # ---------------------------------------------------------------- checks
    failures: list[str] = []

    counts = hmm.groupby("held_out_patient").size()
    seen_in = (
        pd.concat([pd.DataFrame({"p": pd.read_csv(f)["held_out_patient"].unique(),
                                 "shard": f.stem}) for f in files],
                  ignore_index=True)
        .groupby("p")["shard"].nunique()
    )
    duplicates = sorted(seen_in[seen_in > 1].index.tolist())

    n_patients = int(hmm["held_out_patient"].nunique())
    if n_patients != args.expect_patients:
        failures.append(
            f"expected {args.expect_patients:,} held-out patients, found {n_patients:,}")

    if duplicates:
        failures.append(
            f"{len(duplicates)} patient(s) appear in more than one shard, "
            f"first few: {duplicates[:10]}")

    if len(hmm) != args.expect_predictions:
        failures.append(
            f"expected {args.expect_predictions:,} HMM prediction rows, found {len(hmm):,}")

    train_counts = sorted(hmm["n_train_patients"].unique().tolist())
    if train_counts != [EXPECT_TRAIN_PATIENTS]:
        failures.append(
            f"every fold should train on {EXPECT_TRAIN_PATIENTS} patients; "
            f"found {train_counts}")

    prob_sums = hmm[loo.PROB_COLS].sum(axis=1).to_numpy(dtype=float)
    bad = int((~np.isclose(prob_sums, 1.0, atol=1e-6)).sum())
    if bad:
        failures.append(
            f"{bad:,} HMM rows have probabilities that do not sum to 1 "
            f"(worst deviation {np.abs(prob_sums - 1).max():.2e})")

    if hmm.duplicated(["held_out_patient", "visit_index"]).any():
        n = int(hmm.duplicated(["held_out_patient", "visit_index"]).sum())
        failures.append(f"{n:,} duplicated (patient, visit) prediction rows")

    # Which patients are missing, if the cohort can be rebuilt here.
    missing: list = []
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if panel_path.exists():
        states = [cfg.subtype.labels[n] for n in loo.STATE_NAMES]
        data = loo.prepare(pd.read_csv(panel_path, low_memory=False), states)
        missing = sorted(set(data.patients) - set(hmm["held_out_patient"]))
        if missing:
            failures.append(
                f"{len(missing)} patient(s) missing from the shards, "
                f"first few: {missing[:10]}")
    else:
        print(f"\nnote: {panel_path} not present, so the missing-patient check "
              f"was skipped")

    # ----------------------------------------------------------- convergence
    by_fold = hmm.groupby("held_out_patient")["train_converged"].first()
    n_conv = int(by_fold.sum())
    n_fail = int(len(by_fold) - n_conv)

    print(f"\npatients            {n_patients:,}")
    print(f"HMM prediction rows {len(hmm):,}")
    print(f"predictions/patient median {int(counts.median())}, "
          f"range {int(counts.min())}-{int(counts.max())}")
    print(f"latent states       {sorted(hmm['n_hidden'].dropna().unique().tolist())}")
    print(f"converged folds     {n_conv:,} ({n_conv / max(len(by_fold), 1):.1%})")
    print(f"non-converged folds {n_fail:,}")
    print(f"duplicate patients  {len(duplicates)}")
    print(f"missing patients    {len(missing)}")
    print(f"models present      {sorted(allrows['model'].unique().tolist())}")

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nNothing written. Fix the shards and rerun; do not merge partial "
              "results into a pooled figure.")
        return 1

    base.mkdir(parents=True, exist_ok=True)
    dest = base / "loo_predictions.csv"
    hmm.to_csv(dest, index=False)
    allrows.to_csv(base / "loo_predictions_all_models.csv", index=False)
    print(f"\nall checks passed")
    print(f"wrote {dest}")
    print(f"wrote {base / 'loo_predictions_all_models.csv'}  (HMM + baselines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
