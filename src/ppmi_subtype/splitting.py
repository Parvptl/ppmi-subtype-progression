"""Patient-level train/test splitting for held-out validation.

The split is at the level of the *patient*, never the visit. A patient
contributes several transition pairs; if those were split independently the
same person would appear on both sides and the held-out estimate would be
optimistic. The manifest of which patient went where is returned alongside the
split so a run can be reproduced exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_patients(
    panel: pd.DataFrame,
    train_fraction: float,
    seed: int,
    stratify_by: str | None = "baseline_subtype",
    label_col: str = "subtype",
    time_col: str = "years_from_baseline",
) -> pd.DataFrame:
    """Assign every patient with at least one labelled visit to train or test.

    Stratification is on the patient's *baseline* subtype --- their state at
    their first labelled visit --- so the two sides carry a similar mix of
    starting phenotypes. Without it, a chance imbalance in starting state would
    make the held-out comparison partly a comparison of populations.

    Returns a frame with columns ``PATNO``, ``split`` and ``baseline_subtype``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must lie in (0, 1); got {train_fraction}")

    labelled = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])
    first = labelled.groupby("PATNO").head(1)[["PATNO", label_col]]
    first = first.rename(columns={label_col: "baseline_subtype"})

    rng = np.random.default_rng(seed)
    assignments = []

    groups = (
        [(v, g) for v, g in first.groupby("baseline_subtype")]
        if stratify_by
        else [(None, first)]
    )

    for _, group in groups:
        patients = group["PATNO"].to_numpy()
        # Sort before shuffling so the result depends only on the seed, not on
        # the order rows happened to arrive in.
        patients = np.sort(patients)
        rng.shuffle(patients)
        n_train = int(round(train_fraction * len(patients)))
        for k, p in enumerate(patients):
            assignments.append({"PATNO": p, "split": "train" if k < n_train else "test"})

    manifest = pd.DataFrame(assignments).merge(first, on="PATNO", how="left")
    return manifest.sort_values("PATNO").reset_index(drop=True)


def apply_split(pairs: pd.DataFrame, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select the transition pairs belonging to one side of the split."""
    keep = set(manifest.loc[manifest["split"] == split, "PATNO"])
    return pairs[pairs["PATNO"].isin(keep)].reset_index(drop=True)


def split_summary(manifest: pd.DataFrame, names: dict | None = None) -> pd.DataFrame:
    """Patient counts and baseline-subtype composition on each side."""
    out = pd.crosstab(manifest["split"], manifest["baseline_subtype"])
    if names:
        out.columns = [names.get(c, c) for c in out.columns]
    out["total"] = out.sum(axis=1)
    props = out.drop(columns="total").div(out["total"], axis=0)
    props.columns = [f"prop_{c}" for c in props.columns]
    return pd.concat([out, props.round(3)], axis=1)
