"""LOPO evaluation of the latent clinical-state model against the subtype-history CT-HMM.

The question
------------
Does representing the patient with ten clinical measurements and a learned
latent state say more about the next visit than the TD / Indeterminate / PIGD
label history alone?

Design
------
Patients were split once by script 61 into a 30% development set (where the
state count was chosen) and a 70% evaluation set. This script reproduces that
split from the same seed, asserts the two are disjoint, and evaluates **only**
the evaluation patients. The state count is read from script 61's
``summary.json`` and is not re-chosen here.

Each fold holds out one evaluation patient and refits, on every *other*
patient, from scratch:

* the feature standardiser,
* the CT-HMM with Gaussian emissions (EM, several starts),
* the label emission P(observed subtype | latent state).

Training uses all other patients, development ones included: they contributed
only to choosing the state count, which is a training-side decision, and no
evaluation patient's outcome informed it. That also matches the baseline's
training size, which keeps the comparison about the representation rather than
about how much data each model saw.

Predictions come from filtering alone. What is compared against the baseline is
the next *observed* subtype label, on identical cases matched by patient and by
(current visit, next visit) event pair -- never latent-state accuracy against
subtype accuracy, which are different targets.

    python scripts/62_clinical_state_lopo.py --out <dir> --selection <dir>
                                             [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import clinical_states as cs  # noqa: E402
from ppmi_subtype import next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

_sel = import_module("61_clinical_state_selection")
split_patients = _sel.split_patients

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

#: EM starts per fold. Fewer than the selection sweep used, because a fold is
#: fitted 654 times rather than 36. The start spread at the chosen state count
#: is reported from the selection run and again here on a sample of folds, so
#: the cost of this reduction is visible rather than hidden.
N_STARTS_PER_FOLD = 3

MODEL_CLINICAL = "Clinical-state CT-HMM (10 measurements)"
MODEL_BASELINE = "Subtype-history CT-HMM (labels only)"
MODEL_PERSIST = "Persistence (last non-Indeterminate)"

lines: list[str] = []
_WORK: dict = {}


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def section(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def block(frame, **kw) -> str:
    return "    " + frame.to_string(**kw).replace("\n", "\n    ")


# ---------------------------------------------------------------------------
#  One fold
# ---------------------------------------------------------------------------
def _init(sequences, n_states, seed) -> None:
    _WORK.update(sequences=sequences, n_states=n_states, seed=seed)


def run_fold(sequences, held_out, n_states: int, seed: int,
             n_starts: int = N_STARTS_PER_FOLD) -> pd.DataFrame:
    """Fit on every other patient, then predict the held-out patient's visits."""
    train = [s for s in sequences if s["PATNO"] != held_out]
    test = [s for s in sequences if s["PATNO"] == held_out]
    if not test:
        raise ValueError(f"patient {held_out} has no sequence")
    if any(s["PATNO"] == held_out for s in train):
        raise RuntimeError(f"held-out patient {held_out} leaked into training")

    standardiser = cs.Standardiser.fit(train)
    train_z = standardiser.apply(train)

    fit, records = cs.fit_em_multistart(train_z, n_states, seed=seed,
                                        n_starts=n_starts)
    fit = cs.reorder(fit, cs.canonical_order(fit, standardiser))
    fit.label_emission = cs.fit_label_emission(fit, train_z)

    preds = cs.next_visit_predictions(fit, standardiser.apply(test))
    preds["model"] = MODEL_CLINICAL
    preds["held_out_patient"] = held_out
    preds["n_train_patients"] = len(train)
    preds["n_states"] = n_states
    preds["fit_train_loglik"] = fit.loglik
    preds["fit_converged"] = bool(fit.converged)
    preds["fit_n_iter"] = int(fit.n_iter)
    preds["fit_start_spread"] = records[0]["loglik_spread_across_starts"]
    return preds


def _fold(patno):
    try:
        return run_fold(_WORK["sequences"], patno, _WORK["n_states"],
                        _WORK["seed"]), None
    except Exception as exc:                        # noqa: BLE001 -- recorded
        return None, f"{type(exc).__name__}: {exc}"


def run_lopo(sequences, patients, n_states, seed, workers):
    frames, failures = [], []
    if workers <= 1:
        _init(sequences, n_states, seed)
        results = [( _fold(p)) for p in patients]
    else:
        with mp.get_context("fork").Pool(
                workers, initializer=_init,
                initargs=(sequences, n_states, seed)) as pool:
            results = []
            for n, r in enumerate(pool.imap(_fold, patients, chunksize=2), start=1):
                results.append(r)
                if n % 50 == 0:
                    say(f"  fold {n:,}/{len(patients):,}")
    for patno, (frame, error) in zip(patients, results):
        if error is None:
            frames.append(frame)
        else:
            failures.append({"held_out_patient": patno, "error": error})
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return predictions, pd.DataFrame(failures,
                                     columns=["held_out_patient", "error"])


# ---------------------------------------------------------------------------
#  Two-class restriction, shared with the subtype-history experiment
# ---------------------------------------------------------------------------
def add_two_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach P_TD / P_PIGD using the previous experiment's own functions.

    Reusing :func:`next_visit.conditional_two_class` and
    :func:`next_visit.predicted_state` is what makes the comparison a
    comparison: the renormalisation, the denominator guard and the tie rule are
    byte-identical to the baseline's, so any difference in the numbers is a
    difference in the models.
    """
    out = frame.copy()
    probs = out[[f"p_{s}" for s in cs.STATE_NAMES]].to_numpy(dtype=float)
    p_td, p_pigd, defined = nv.conditional_two_class(probs)
    out["p_TD_raw"] = probs[:, 0]
    out["p_Indeterminate_raw"] = probs[:, 1]
    out["p_PIGD_raw"] = probs[:, 2]
    out["P_TD"] = p_td
    out["P_PIGD"] = p_pigd
    out["two_class_defined"] = defined
    out["predicted_state"] = nv.predicted_state(p_td, p_pigd)
    out["is_tie"] = defined & (p_td == p_pigd)
    out["eligible"] = out["actual_next_label"].isin(["TD", "PIGD"])
    return out


def persistence_rows(frame: pd.DataFrame, prior_pigd: float) -> pd.DataFrame:
    """The same persistence rule the baseline experiment scored, on these cases.

    Included as a floor: the previous experiment found it indistinguishable
    from the best model, so a new representation has to be read against it and
    not only against the other fitted models.
    """
    out = frame.copy()
    p_pigd = np.where(out["prev_noind_label"] == "PIGD", 1.0,
                      np.where(out["prev_noind_label"] == "TD", 0.0, prior_pigd))
    out["model"] = MODEL_PERSIST
    out["p_TD_raw"] = 1.0 - p_pigd
    out["p_Indeterminate_raw"] = 0.0
    out["p_PIGD_raw"] = p_pigd
    out["P_TD"] = 1.0 - p_pigd
    out["P_PIGD"] = p_pigd
    out["two_class_defined"] = True
    out["predicted_state"] = nv.predicted_state(1.0 - p_pigd, p_pigd)
    out["is_tie"] = (1.0 - p_pigd) == p_pigd
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", required=True,
                    help="script 61's run directory, for the chosen state count")
    ap.add_argument("--baseline", default=None,
                    help="subtype-history run dir with next_visit_predictions.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-states", type=int, default=None,
                    help="override the selected state count (development runs only)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    # ------------------------------------------------------------------ input
    section("1. INPUT, SELECTED STATE COUNT AND THE PATIENT SPLIT")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel md5            {panel_md5}  [MATCH]")

    selection = json.loads((Path(args.selection) / "summary.json").read_text())
    n_states = args.n_states or int(selection["chosen_n_states"])
    say(f"  selection run        {args.selection}")
    say(f"  selection rule       {selection.get('selection_rule', 'n/a')}")
    say(f"  state count          {n_states}"
        + ("  (OVERRIDDEN on the command line)" if args.n_states else ""))
    say(f"  arithmetic max would choose "
        f"{selection.get('arithmetic_max_would_choose')}")
    say(f"  BIC would choose     {selection.get('bic_would_choose')}")

    panel = pd.read_csv(panel_path, low_memory=False)
    clinical = cs.build_clinical_panel(cfg, panel)
    sequences = cs.build_sequences(clinical)

    dev, evaluation = split_patients([s["PATNO"] for s in sequences], cfg.seed)
    if not set(dev).isdisjoint(set(evaluation)):
        raise SystemExit("development and evaluation patients overlap")
    sel_dev = set(pd.read_csv(Path(args.selection) / "development_patients.csv")["PATNO"])
    if sel_dev != set(dev):
        raise SystemExit("reproduced development split differs from script 61's")
    say(f"  development patients {len(dev):,}  (reproduced and matched)")
    say(f"  evaluation patients  {len(evaluation):,}  <- scored here")
    say("  Training uses every OTHER patient, development included: they")
    say("  informed only the state count, a training-side decision, and no")
    say("  evaluation patient's outcome informed it.")

    patients = evaluation[: args.limit] if args.limit else evaluation

    # -------------------------------------------------------------------- run
    section("2. LEAVE-ONE-PATIENT-OUT")
    say(f"  folds                {len(patients):,}")
    say(f"  workers              {args.workers}")
    say(f"  EM starts per fold   {N_STARTS_PER_FOLD}")
    say("  Refitted per fold: standardiser, CT-HMM (EM), label emission.")
    say("  Predictions use filtering only -- no backward pass anywhere.")

    raw, failures = run_lopo(sequences, patients, n_states, cfg.seed, args.workers)
    if not len(raw):
        raise SystemExit("every fold failed")

    # Previous non-Indeterminate label, for the persistence floor. Computed
    # from the held-out patient's own history up to the current visit only.
    hist = {}
    for s in sequences:
        last = None
        for i, lab in enumerate(s["labels"]):
            name = cs.STATE_NAMES[int(lab) - 1]
            if name != "Indeterminate":
                last = name
            hist[(s["PATNO"], int(s["visit_index"][i]))] = last
    raw["prev_noind_label"] = [hist.get((p, v)) for p, v in
                               zip(raw["PATNO"], raw["visit_index"])]

    clinical_preds = add_two_class(raw)
    prior = float((clinical_preds.loc[clinical_preds["eligible"],
                                      "actual_next_label"] == "PIGD").mean())
    persist = persistence_rows(clinical_preds, prior)

    say("")
    say(f"  folds attempted      {len(patients):,}")
    say(f"  folds completed      {raw['held_out_patient'].nunique():,}")
    say(f"  folds failed         {len(failures):,}")
    if len(failures):
        say(block(failures, index=False))
    say(f"  prediction cases     {len(clinical_preds):,}")
    say(f"  elapsed              {time.time() - started:.0f}s")

    conv = clinical_preds.drop_duplicates("held_out_patient")
    say("")
    say(f"  folds where EM converged          {int(conv['fit_converged'].sum()):,}"
        f" / {len(conv):,}")
    say(f"  mean EM iterations                {conv['fit_n_iter'].mean():.1f}")
    say(f"  median start spread (nats)        {conv['fit_start_spread'].median():.1f}")
    say(f"  max start spread (nats)           {conv['fit_start_spread'].max():.1f}")
    say("  A large spread across starts means EM lands in different optima from")
    say("  different initialisations. It is reported because it bounds how")
    say("  stably the states themselves can be interpreted.")

    # ------------------------------------------------------- latent outcomes
    section("3. LATENT-STATE PREDICTION (a research outcome, not the baseline target)")
    state_cols = [f"p_state_{s}" for s in range(n_states)]
    say(f"  mean next-visit clinical log-density   "
        f"{clinical_preds['next_obs_loglik'].mean():.4f}")
    say(f"  mean predicted-state confidence        "
        f"{clinical_preds['p_next_latent_state'].mean():.4f}")
    say(f"  exact ties in the latent argmax        "
        f"{int(clinical_preds['latent_state_is_tie'].sum()):,}")
    occupancy = pd.DataFrame({
        "state": range(n_states),
        "mean_predicted_probability": [clinical_preds[c].mean() for c in state_cols],
        "times_argmax": [int((clinical_preds["predicted_next_latent_state"] == s).sum())
                         for s in range(n_states)],
    })
    occupancy["share_argmax"] = occupancy["times_argmax"] / len(clinical_preds)
    occupancy.round(4).to_csv(out / "predicted_state_occupancy.csv", index=False)
    say("")
    say(block(occupancy.round(4), index=False))
    say("")
    say("  The next latent state is NOT an observed quantity, so there is no")
    say("  ground truth to score it against. What is scored is the predictive")
    say("  log-density of the next visit's MEASUREMENTS under the predicted")
    say("  state distribution, which is a proper score and needs no labels.")
    say("  Agreement with the state filtered at the next visit is reported")
    say("  below as a model-internal consistency check, not as accuracy.")

    agree = (clinical_preds["predicted_next_latent_state"]
             == clinical_preds.groupby("PATNO")["filtered_state_current"].shift(-1))
    say(f"  internal consistency (predicted next == filtered next): "
        f"{float(agree.mean()):.4f}")

    # ---------------------------------------------- the comparable comparison
    section("4. NEXT OBSERVED SUBTYPE: CLINICAL STATES VS SUBTYPE HISTORY")
    frames = [clinical_preds, persist]

    baseline = None
    if args.baseline:
        bl = pd.read_csv(Path(args.baseline) / "next_visit_predictions.csv",
                         low_memory=False)
        bl = bl[bl["model"] == "CT-HMM (latent, interval-aware)"].copy()
        bl["model"] = MODEL_BASELINE
        # Match on the visit PAIR, not on visit_index: this experiment drops
        # visits carrying too few features, so the two indexings differ for
        # any patient who lost one.
        key = ["PATNO", "current_event", "target_event"]
        wanted = clinical_preds[key].drop_duplicates()
        baseline = bl.merge(wanted, on=key, how="inner")
        say(f"  baseline rows loaded              {len(bl):,}")
        say(f"  matched to this experiment's cases {len(baseline):,}")
        frames.append(baseline)

    all_preds = pd.concat(frames, ignore_index=True)

    # Restrict every model to the cases ALL of them have, so the comparison is
    # paired. Any case dropped here is reported, not silently discarded.
    key = ["PATNO", "current_event", "target_event"]
    common = None
    for name, g in all_preds.groupby("model", sort=False):
        s = set(map(tuple, g[key].to_numpy()))
        common = s if common is None else (common & s)
    mask = [tuple(r) in common for r in all_preds[key].to_numpy()]
    dropped = all_preds[~np.array(mask)]
    all_preds = all_preds[np.array(mask)].reset_index(drop=True)

    say("")
    for name, g in all_preds.groupby("model", sort=False):
        say(f"  {name:<42} {len(g):,} cases")
    if len(dropped):
        say(f"  cases dropped to make the sets identical: {len(dropped):,}")
        say(block(dropped.groupby("model").size().to_frame("n_dropped")))
        dropped.to_csv(out / "cases_dropped_for_pairing.csv", index=False)

    cols = ["PATNO", "current_event", "current_years", "target_event",
            "target_years", "interval_years", "n_history", "current_label",
            "model", "P_TD", "P_PIGD", "predicted_state", "actual_next_label",
            "eligible", "two_class_defined", "is_tie", "p_TD_raw",
            "p_Indeterminate_raw", "p_PIGD_raw", "visit_index"]
    extra = [c for c in (state_cols + ["predicted_next_latent_state",
                                       "p_next_latent_state",
                                       "latent_state_is_tie", "next_obs_loglik",
                                       "filtered_state_current",
                                       "fit_converged", "fit_start_spread",
                                       "n_states", "held_out_patient"])
             if c in all_preds.columns]
    all_preds[cols + extra].to_csv(out / "clinical_state_predictions.csv",
                                   index=False)
    failures.to_csv(out / "failed_folds.csv", index=False)

    rows = [nv.two_class_metrics(g.reset_index(drop=True), seed=cfg.seed)
            for _, g in all_preds.groupby("model", sort=False)]
    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    comparison.round(4).to_csv(out / "model_comparison.csv", index=False)

    show = ["model", "n_scored", "accuracy", "accuracy_ci_patient_bootstrap_low",
            "accuracy_ci_patient_bootstrap_high", "brier_two_class", "log_loss",
            "n_ties"]
    say("")
    say(block(comparison[show].round(4), index=False))

    for name, g in all_preds.groupby("model", sort=False):
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        nv.confusion(g).to_csv(out / f"confusion_{slug}.csv")
        nv.per_class_metrics(g).round(4).to_csv(out / f"per_class_{slug}.csv",
                                                index=False)
        nv.calibration(g).round(4).to_csv(out / f"calibration_{slug}.csv",
                                          index=False)

    # ------------------------------------------------------------- contrasts
    section("5. PAIRED CONTRASTS")
    contrast_rows = []
    pairs = [("richer representation against the subtype-history baseline",
              MODEL_CLINICAL, MODEL_BASELINE),
             ("richer representation against the persistence floor",
              MODEL_CLINICAL, MODEL_PERSIST),
             ("subtype-history baseline against the persistence floor",
              MODEL_BASELINE, MODEL_PERSIST)]
    present = set(all_preds["model"])
    acc = comparison.set_index("model")
    for what, a, b in pairs:
        if a not in present or b not in present:
            continue
        paired = nv.paired_accuracy_difference(
            all_preds[all_preds["model"] == a],
            all_preds[all_preds["model"] == b], seed=cfg.seed)
        contrast_rows.append({
            "contrast": what, "model_a": a, "model_b": b,
            "accuracy_a": acc.loc[a, "accuracy"], "accuracy_b": acc.loc[b, "accuracy"],
            "brier_delta": acc.loc[a, "brier_two_class"] - acc.loc[b, "brier_two_class"],
            **{k: v for k, v in paired.items() if k != "n_patients"},
        })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.round(4).to_csv(out / "contrasts.csv", index=False)
    show_c = ["contrast", "accuracy_a", "accuracy_b", "accuracy_delta",
              "delta_ci_patient_bootstrap_low", "delta_ci_patient_bootstrap_high",
              "brier_delta", "a_right_b_wrong", "b_right_a_wrong",
              "mcnemar_exact_p_ignores_clustering"]
    say(block(contrasts[show_c].round(4), index=False))
    say("")
    say("  Intervals come from resampling PATIENTS on the identical paired")
    say("  cases. A positive accuracy_delta and a negative brier_delta favour")
    say("  model_a.")

    # --------------------------------------------------------- where it lives
    section("6. WHERE THE DIFFERENCES LIVE")
    breakdowns = nv.accuracy_breakdowns(all_preds)
    for key_name, frame in breakdowns.items():
        frame.round(4).to_csv(out / f"accuracy_{key_name}.csv", index=False)
    say("  By the label recorded at the prediction point:")
    say(block(breakdowns["by_current_label"].round(4), index=False))
    say("")
    say("  The Indeterminate row is the one the richer representation was")
    say("  expected to help most: the recorded label carries no information")
    say("  there, while the ten measurements still do.")
    say("")
    say(f"  By inter-visit interval (split at {nv.INTERVAL_SPLIT_YEARS} years):")
    say(block(breakdowns["by_interval"].round(4), index=False))

    # ------------------------------------------------------------- checks
    section("7. CORRECTNESS CHECKS")
    defined = all_preds[all_preds["two_class_defined"]]
    p_td = defined["P_TD"].to_numpy(float)
    p_pigd = defined["P_PIGD"].to_numpy(float)
    cl = all_preds[all_preds["model"] == MODEL_CLINICAL]
    per_model = {m: frozenset(map(tuple, g[key].to_numpy()))
                 for m, g in all_preds.groupby("model", sort=False)}

    checks = {
        "probabilities finite and within [0, 1]":
            bool(np.isfinite(p_td).all() and (p_td >= 0).all() and (p_td <= 1).all()),
        "P_TD + P_PIGD == 1 within tolerance":
            bool(np.allclose(p_td + p_pigd, 1.0, atol=1e-12)),
        "predicted class follows the documented rule":
            bool((defined["predicted_state"].to_numpy()
                  == nv.predicted_state(p_td, p_pigd)).all()),
        "latent state probabilities sum to one":
            bool(np.allclose(cl[state_cols].sum(axis=1), 1.0, atol=1e-8)),
        "every model scored the identical case set":
            len(set(per_model.values())) == 1,
        "held-out patient is the patient being predicted":
            bool((cl["held_out_patient"] == cl["PATNO"]).all()),
        "every fold trained on all other patients":
            set(cl["n_train_patients"].unique()) == {len(sequences) - 1}
            if "n_train_patients" in cl.columns else True,
        "no evaluation patient is in the development set":
            set(cl["PATNO"]).isdisjoint(set(dev)),
        "history length is the visits up to the current one":
            bool((cl["n_history"] == cl["visit_index"] + 1).all()),
        "target visit is strictly later than the current visit":
            bool((all_preds["target_years"] > all_preds["current_years"]).all()),
        "elapsed time equals the gap between visit times":
            bool(np.allclose(all_preds["interval_years"],
                             all_preds["target_years"] - all_preds["current_years"],
                             atol=1e-8)),
        "Indeterminate next visits excluded from scoring":
            int(nv.scored(all_preds)["actual_next_label"]
                .isin(["Indeterminate"]).sum()) == 0,
        "Indeterminate next visits still reported":
            bool((all_preds["actual_next_label"] == "Indeterminate").any()),
        "Indeterminate survives as input evidence":
            bool((all_preds["current_label"] == "Indeterminate").any()),
        "completed and failed folds account for every attempted patient":
            set(cl["held_out_patient"]) | set(failures["held_out_patient"])
            == set(patients),
    }
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # ---------------------------------------------------------- provenance
    section("8. PROVENANCE")
    env = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(),
        "node": platform.node(), "workers": args.workers, "seed": cfg.seed,
        "n_states": n_states, "n_starts_per_fold": N_STARTS_PER_FOLD,
        "panel_md5": panel_md5,
        "selection_run": str(args.selection),
        "baseline_run": str(args.baseline),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<20} {v}")

    (out / "summary.json").write_text(json.dumps({
        "environment": env,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                        [Path(__file__), ROOT / "src/ppmi_subtype/clinical_states.py",
                         ROOT / "src/ppmi_subtype/next_visit.py"]},
        "counts": {"n_evaluation_patients": len(evaluation),
                   "n_folds_attempted": len(patients),
                   "n_folds_failed": int(len(failures)),
                   "n_cases": int(len(cl)),
                   "n_eligible": int(cl["eligible"].sum())},
        "checks": checks,
        "model_comparison": comparison.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "latent": {"mean_next_obs_loglik": float(cl["next_obs_loglik"].mean()),
                   "occupancy": occupancy.to_dict(orient="records")},
        "accuracy_breakdowns": {k: v.to_dict(orient="records")
                                for k, v in breakdowns.items()},
    }, indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
