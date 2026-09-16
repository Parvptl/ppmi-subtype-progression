"""One shard of the patient-level leave-one-out run for the 5-hidden-state HMM.

Step 2 keeps the original subtype thresholds and the original three observed
labels, and gives the latent chain five states instead of three. The emission
matrix is therefore 5x3: each latent state emits a distribution over the three
labels, and several latent states may favour the same label. No structure is
imposed on which label a latent state should prefer --- that is the thing the
experiment is meant to find out.

The emission floor is switched off. It constrains P(label i | state i), and
with five states against three labels there is no such pairing; any
generalisation of it would amount to declaring in advance which label each
extra state stands for.

Sharding
--------
Each array task takes a contiguous block of the patient list and holds out only
those patients. Sharding divides the *work*, not the data: every fold still
trains on all 933 other patients in the cohort, exactly as an unsharded run
would, because run_fold derives its training set from the full LOOData.

    SLURM_ARRAY_TASK_ID   which shard this process is        (default 0)
    NUM_SHARDS            how many shards in total           (default 20)

    python scripts/10_loo_validation_5hidden_shard.py [--limit N] [--dry-run]

Writes outputs/loo_5hidden/shards/shard_XX.csv, with its own checkpoint so
concurrent shards never touch each other's files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import loo  # noqa: E402
from ppmi_subtype.config import load_config, project_root  # noqa: E402

N_HIDDEN = 5
EMISSION_FLOOR = None       # undefined for a rectangular emission matrix
OUTPUT_SUBDIR = "loo_5hidden"


def shard_of(patients: list, shard: int, n_shards: int) -> list:
    """Contiguous block `shard` of `patients`, split as evenly as possible.

    Deterministic given the same patient list: every patient lands in exactly
    one shard and the blocks differ in size by at most one.
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for {n_shards} shards")
    blocks = np.array_split(np.arange(len(patients)), n_shards)
    return [patients[i] for i in blocks[shard]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None,
                    help="override SLURM_ARRAY_TASK_ID")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="override NUM_SHARDS")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N patients of this shard, for testing")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the shard assignment and exit without fitting")
    args = ap.parse_args()

    shard = args.shard if args.shard is not None else int(
        os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_shards = args.num_shards if args.num_shards is not None else int(
        os.environ.get("NUM_SHARDS", "20"))

    # Read the baseline config by explicit path rather than through the
    # PPMI_CONFIG environment variable: Step 2 must use the original
    # thresholds, and a variable left set from the Step 1 experiment would
    # otherwise silently relabel the data.
    cfg = load_config(project_root() / "config" / "config.yaml")
    s = cfg.subtype
    if (s.td_threshold, s.pigd_threshold, s.band_boundaries) != (1.15, 0.90, "outer"):
        raise SystemExit(
            "expected the original thresholds (PIGD <= 0.90, TD >= 1.15); got "
            f"{s.pigd_threshold}/{s.td_threshold} with band_boundaries="
            f"{s.band_boundaries!r}"
        )

    out_dir = cfg.path("outputs") / OUTPUT_SUBDIR / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    states = [s.labels[name] for name in loo.STATE_NAMES]
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    data = loo.prepare(panel, states)

    # Sort so the assignment depends only on the patient identifiers, not on
    # the order rows happened to arrive in.
    patients = sorted(data.patients)
    mine = shard_of(patients, shard, n_shards)
    if args.limit:
        mine = mine[:args.limit]

    print(f"shard {shard} of {n_shards}")
    print(f"  cohort                {len(patients):,} patients, "
          f"{data.n_transitions:,} transitions")
    print(f"  this shard            {len(mine):,} patients "
          f"({mine[0]} .. {mine[-1]})")
    print(f"  latent states         {N_HIDDEN}   observed labels {len(states)}")
    print(f"  emission floor        {EMISSION_FLOOR}")
    print(f"  restarts / seed       {cfg.model['n_restarts']} / {cfg.seed}")
    print(f"  thresholds            PIGD <= {s.pigd_threshold}, TD >= {s.td_threshold}")

    if args.dry_run:
        print("  dry run, nothing fitted")
        return 0

    checkpoint = out_dir / f"_checkpoint_shard_{shard:02d}.csv"
    result = loo.run_loo(
        data,
        seed=cfg.seed,
        n_restarts=cfg.model["n_restarts"],
        emission_floor=EMISSION_FLOOR,
        checkpoint=checkpoint,
        checkpoint_every=5,
        with_baselines=True,
        progress_every=5,
        patients_subset=mine,
        n_hidden=N_HIDDEN,
    )

    held = set(result["held_out_patient"].unique())
    if held != set(mine):
        raise SystemExit(
            f"shard {shard} produced {len(held)} held-out patients, expected "
            f"{len(mine)}"
        )

    hmm_rows = result[result["model"] == "HMM"]
    n_conv = int(hmm_rows.groupby("held_out_patient")["train_converged"].first().sum())
    print(f"\n  folds                 {len(held):,}")
    print(f"  HMM prediction rows   {len(hmm_rows):,}")
    print(f"  converged folds       {n_conv:,} of {len(held):,}")
    print(f"  train patients / fold {sorted(hmm_rows['n_train_patients'].unique())}")

    dest = out_dir / f"shard_{shard:02d}.csv"
    result.to_csv(dest, index=False)
    print(f"\nwrote {dest}")

    checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
