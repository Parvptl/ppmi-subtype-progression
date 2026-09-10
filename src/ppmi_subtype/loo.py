"""Patient-level leave-one-out validation of the progression models.

What this measures
------------------
For each patient in turn, the model is refitted from scratch on every *other*
patient, and then used to predict that patient's recorded subtype one visit at a
time. To predict visit ``k`` the model may use visits ``0 .. k-1`` of the
held-out patient as history; it may never use visit ``k`` or anything after it,
and no visit of the held-out patient is used in fitting.

The target is the patient's **next recorded subtype label**, produced by this
project's own classification rule. It is not an independently verified clinical
diagnosis, and the label itself carries measurement variability, so accuracy
here is accuracy at reproducing a noisy record rather than at predicting disease
progression.

Design notes
------------
* The prediction machinery is :func:`ppmi_subtype.hmm.predictive_distributions`
  unchanged --- the same function the existing train/test evaluation uses --- so
  the two evaluations differ only in how patients are split.
* The elapsed interval to the next visit is an *input* to the prediction, for
  the HMM and the continuous-time chain alike. The model is told when the next
  visit happens, not what is recorded at it. That matches the existing
  evaluation and is the usual convention for panel data, but it is an assumption
  worth stating.
* Every fold refits from the same seed, so the run is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import ctmc, hmm
from .transitions import consecutive_pairs

STATE_NAMES = ["TD", "Indeterminate", "PIGD"]
PROB_COLS = [f"p_{s}" for s in STATE_NAMES]


class LeakageError(RuntimeError):
    """Raised when a held-out patient is found in the training data."""


# ---------------------------------------------------------------------------
#  Fold construction and leakage checking
# ---------------------------------------------------------------------------
@dataclass
class LOOData:
    """Everything the fold loop needs, prepared once."""

    sequences: list[dict]
    pairs: pd.DataFrame
    states: list
    meta: pd.DataFrame          # per-visit metadata, indexed by (PATNO, visit index)

    @property
    def patients(self) -> list:
        return [s["PATNO"] for s in self.sequences]

    @property
    def n_transitions(self) -> int:
        return int(sum(len(s["gaps"]) for s in self.sequences))


def prepare(panel: pd.DataFrame, states: list) -> LOOData:
    """Build sequences, pairs and per-visit metadata from the labelled panel."""
    labelled = panel[panel["subtype"].notna()].sort_values(
        ["PATNO", "years_from_baseline"]
    )
    sequences = hmm.build_sequences(labelled, states)
    keep = {s["PATNO"] for s in sequences}

    followed = labelled[labelled["PATNO"].isin(keep)].copy()
    followed["visit_index_labelled"] = followed.groupby("PATNO").cumcount()

    meta = followed[
        ["PATNO", "visit_index_labelled", "EVENT_ID", "years_from_baseline", "subtype"]
    ].rename(columns={"visit_index_labelled": "visit_index"})

    return LOOData(
        sequences=sequences,
        pairs=consecutive_pairs(followed),
        states=list(states),
        meta=meta,
    )


def assert_no_leakage(held_out, train_sequences: list[dict], train_pairs: pd.DataFrame) -> None:
    """Fail loudly if any trace of the held-out patient is in the training data."""
    train_ids = {s["PATNO"] for s in train_sequences}
    if held_out in train_ids:
        raise LeakageError(f"held-out patient {held_out} is among the training sequences")
    if held_out in set(train_pairs["PATNO"]):
        raise LeakageError(f"held-out patient {held_out} has transitions in the training pairs")


# ---------------------------------------------------------------------------
#  One fold
# ---------------------------------------------------------------------------
def run_fold(
    data: LOOData,
    held_out,
    seed: int,
    n_restarts: int,
    emission_floor: float,
    with_baselines: bool = True,
) -> pd.DataFrame:
    """Fit on every other patient, then predict this patient's visits in order."""
    train_sequences = [s for s in data.sequences if s["PATNO"] != held_out]
    train_pairs = data.pairs[data.pairs["PATNO"] != held_out]
    assert_no_leakage(held_out, train_sequences, train_pairs)

    test_sequence = next(s for s in data.sequences if s["PATNO"] == held_out)
    test_pairs = data.pairs[data.pairs["PATNO"] == held_out].reset_index(drop=True)

    fit = hmm.fit_hmm(
        train_sequences,
        data.states,
        state_names=STATE_NAMES,
        seed=seed,
        n_restarts=n_restarts,
        emission_floor=emission_floor,
    )
    preds = hmm.predictive_distributions(fit, [test_sequence])
    preds = preds.rename(columns={"step": "visit_index"})
    preds["model"] = "HMM"
    preds["train_loglik"] = fit.loglik
    preds["train_converged"] = fit.converged
    preds["n_train_patients"] = len(train_sequences)

    frames = [preds]

    if with_baselines:
        index = {s: k for k, s in enumerate(data.states)}
        i = test_pairs["from_state"].map(index).to_numpy(int)

        baselines = {
            "Markov (discrete)": ctmc.fit_dtmc(train_pairs, data.states),
            "Marginal (null)": ctmc.fit_marginal(train_pairs, data.states),
            "CTMC (interval-aware)": ctmc.fit_ctmc(
                train_pairs, data.states, state_names=STATE_NAMES,
                seed=seed, n_restarts=n_restarts,
            ),
        }
        # Persistence carries no free parameters: it always names the current
        # state. Given as a one-hot so it shares the prediction table's schema,
        # but its probabilities are a rule rather than an estimate.
        persistence = np.zeros((len(test_pairs), len(data.states)))
        persistence[np.arange(len(test_pairs)), i] = 1.0
        baselines["Persistence (stay)"] = None

        for name, model in baselines.items():
            probs = (
                persistence if model is None
                else ctmc.predict_pairs(model, test_pairs, data.states)
            )
            frame = test_pairs[["PATNO", "from_state", "to_state", "interval_years"]].copy()
            frame["visit_index"] = np.arange(len(test_pairs))
            for k, col in enumerate(PROB_COLS):
                frame[col] = probs[:, k]
            frame["model"] = name
            frame["train_loglik"] = np.nan
            frame["train_converged"] = True
            frame["n_train_patients"] = len(train_sequences)
            frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["held_out_patient"] = held_out
    return out


# ---------------------------------------------------------------------------
#  The loop, with checkpointing
# ---------------------------------------------------------------------------
def run_loo(
    data: LOOData,
    seed: int,
    n_restarts: int,
    emission_floor: float,
    checkpoint: Path | None = None,
    checkpoint_every: int = 25,
    limit: int | None = None,
    with_baselines: bool = True,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Hold out every patient in turn, resuming from a checkpoint if one exists."""
    patients = data.patients[:limit] if limit else data.patients

    done: set = set()
    collected: list[pd.DataFrame] = []
    if checkpoint is not None and checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        prior = prior[prior["held_out_patient"].isin(patients)]
        if len(prior):
            done = set(prior["held_out_patient"].unique())
            collected.append(prior)
            print(f"  resuming: {len(done):,} folds already complete")

    pending = [p for p in patients if p not in done]
    buffer: list[pd.DataFrame] = []

    for n, patno in enumerate(pending, start=1):
        buffer.append(
            run_fold(data, patno, seed, n_restarts, emission_floor, with_baselines)
        )
        if progress_every and n % progress_every == 0:
            print(f"  fold {n:,}/{len(pending):,}  (patient {patno})", flush=True)

        if checkpoint is not None and n % checkpoint_every == 0:
            collected.extend(buffer)
            pd.concat(collected, ignore_index=True).to_csv(checkpoint, index=False)
            buffer = []

    collected.extend(buffer)
    result = pd.concat(collected, ignore_index=True)
    if checkpoint is not None:
        result.to_csv(checkpoint, index=False)
    return result


# ---------------------------------------------------------------------------
#  Turning raw predictions into something readable
# ---------------------------------------------------------------------------
def annotate(preds: pd.DataFrame, data: LOOData, names: dict) -> pd.DataFrame:
    """Attach names, correctness, confidence and per-visit metadata."""
    out = preds.copy()

    probs = out[PROB_COLS].to_numpy(dtype=float)
    out["predicted_index"] = probs.argmax(axis=1)
    out["predicted"] = [STATE_NAMES[k] for k in out["predicted_index"]]
    out["confidence"] = probs.max(axis=1)

    out["current"] = out["from_state"].map(names)
    out["actual"] = out["to_state"].map(names)
    out["correct"] = out["predicted"] == out["actual"]
    out["transition"] = out["current"] + " -> " + out["actual"]

    # Probability the model gave to the class that actually occurred.
    actual_idx = out["actual"].map({s: k for k, s in enumerate(STATE_NAMES)}).to_numpy(int)
    out["p_actual"] = probs[np.arange(len(out)), actual_idx]

    # History available before the prediction: visits 0 .. visit_index.
    out["n_history"] = out["visit_index"].astype(int) + 1

    current = data.meta.rename(columns={
        "years_from_baseline": "current_years", "EVENT_ID": "current_event"})
    target = data.meta.assign(visit_index=data.meta["visit_index"] - 1).rename(
        columns={"years_from_baseline": "target_years", "EVENT_ID": "target_event"})

    out = out.merge(
        current[["PATNO", "visit_index", "current_years", "current_event"]],
        on=["PATNO", "visit_index"], how="left")
    out = out.merge(
        target[["PATNO", "visit_index", "target_years", "target_event"]],
        on=["PATNO", "visit_index"], how="left")

    out = out.sort_values(["model", "PATNO", "visit_index"]).reset_index(drop=True)
    out["prev_interval_years"] = out.groupby(["model", "PATNO"])["interval_years"].shift()
    return out


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def bootstrap_patient_ci(
    frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI for accuracy, resampling *patients* rather than predictions.

    Predictions are clustered within patients --- a patient with fourteen visits
    contributes thirteen correlated rows --- so an interval that treats them as
    independent is too narrow.
    """
    rng = np.random.default_rng(seed)
    by_patient = {p: g["correct"].to_numpy() for p, g in frame.groupby("PATNO")}
    patients = np.array(list(by_patient.keys()), dtype=object)

    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(patients), size=len(patients), replace=True)
        hits = np.concatenate([by_patient[patients[k]] for k in picked])
        draws[b] = hits.mean()
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, as used elsewhere in this project."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def overall_metrics(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """Headline numbers for one model."""
    n = len(frame)
    correct = int(frame["correct"].sum())
    per_patient = frame.groupby("PATNO")["correct"].mean()
    lo, hi = bootstrap_patient_ci(frame, n_boot=n_boot, seed=seed)
    w_lo, w_hi = wilson_interval(correct, n)

    probs = frame[PROB_COLS].to_numpy(dtype=float)
    actual = frame["actual"].map({s: k for k, s in enumerate(STATE_NAMES)}).to_numpy(int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    p_true = np.clip(probs[np.arange(len(actual)), actual], 1e-12, None)

    return {
        "n_patients": int(frame["PATNO"].nunique()),
        "n_predictions": n,
        "n_correct": correct,
        "accuracy": correct / n,
        "accuracy_ci_patient_bootstrap_low": lo,
        "accuracy_ci_patient_bootstrap_high": hi,
        "accuracy_ci_wilson_low": w_lo,
        "accuracy_ci_wilson_high": w_hi,
        "mean_per_patient_accuracy": float(per_patient.mean()),
        "median_per_patient_accuracy": float(per_patient.median()),
        "sd_per_patient_accuracy": float(per_patient.std()),
        "loglik_per_prediction": float(np.log(p_true).mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "mean_confidence": float(frame["confidence"].mean()),
    }


def confusion(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows are the predicted class, columns the actual class."""
    return (
        pd.crosstab(frame["predicted"], frame["actual"])
        .reindex(index=STATE_NAMES, columns=STATE_NAMES, fill_value=0)
    )


def per_class_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    cm = confusion(frame)
    rows = []
    for state in STATE_NAMES:
        tp = int(cm.loc[state, state])
        predicted = int(cm.loc[state].sum())
        support = int(cm[state].sum())
        precision = tp / predicted if predicted else float("nan")
        recall = tp / support if support else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if predicted and support and (precision + recall) > 0 else float("nan"))
        rows.append({"state": state, "support": support, "n_predicted": predicted,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def transition_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """Accuracy for each observed from -> to combination, rare ones included."""
    rows = []
    for current in STATE_NAMES:
        for actual in STATE_NAMES:
            g = frame[(frame["current"] == current) & (frame["actual"] == actual)]
            rows.append({
                "current": current, "actual": actual,
                "transition": f"{current} -> {actual}",
                "n": len(g),
                "n_correct": int(g["correct"].sum()) if len(g) else 0,
                "accuracy": float(g["correct"].mean()) if len(g) else float("nan"),
                "mean_p_actual": float(g["p_actual"].mean()) if len(g) else float("nan"),
            })
    return pd.DataFrame(rows)


def calibration(frame: pd.DataFrame, min_bin: int = 30) -> pd.DataFrame:
    """Observed accuracy against predicted confidence, with sparse bins merged."""
    edges = [1 / 3, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0001]
    conf = frame["confidence"].to_numpy()

    # Merge upwards until every bin carries enough predictions to be readable.
    while True:
        idx = np.digitize(conf, edges[1:-1], right=False)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        small = [k for k, c in enumerate(counts) if 0 < c < min_bin]
        if not small or len(edges) <= 3:
            break
        k = small[0]
        edges.pop(k + 1 if k + 1 < len(edges) - 1 else k)

    idx = np.digitize(conf, edges[1:-1], right=False)
    rows = []
    for k in range(len(edges) - 1):
        g = frame[idx == k]
        if not len(g):
            continue
        rows.append({
            "bin": f"{edges[k]:.2f}-{edges[k + 1]:.2f}",
            "bin_low": edges[k], "bin_high": edges[k + 1],
            "n": len(g),
            "mean_confidence": float(g["confidence"].mean()),
            "observed_accuracy": float(g["correct"].mean()),
        })
    return pd.DataFrame(rows)


def patient_summary(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("PATNO")
    out = pd.DataFrame({
        "n_predictions": g.size(),
        "n_correct": g["correct"].sum().astype(int),
        "accuracy": g["correct"].mean(),
        "mean_confidence": g["confidence"].mean(),
        "last_followup_years": g["target_years"].max(),
    })
    out["n_labelled_visits"] = out["n_predictions"] + 1
    return out.reset_index()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
