"""Two-hidden-state CT-HMM variants against the existing three-state model.

Four CT-HMM arms, a rule and a null, all refitted from the frozen panel inside
one leave-one-patient-out loop so that nothing differs between them except the
thing under test:

    M1   2 hidden x 2 observed, Indeterminate visits excluded   (5 parameters)
    M1'  the same, without the emission floor                   (5 parameters)
    M2   2 hidden x 3 observed, Indeterminate as an emission     (7 parameters)
    E    3 hidden x 3 observed, the existing model              (12 parameters)
    Persistence (last observed label) and the training label prior

Two evaluations are produced, because the arms do not share an observation
space. The NATIVE one scores each model on its own cases in its own space. The
COMMON one restricts to the target visits M1 can predict and scores every arm
there on {TD, PIGD}, renormalising the three-label models with
next_visit.conditional_two_class. Only the second is a like-for-like accuracy
comparison; the first is reported so the difference in eligible observations
is visible rather than buried.

    python scripts/92_two_state_hmm_comparison.py --out <dir> [--workers N]
                                                  [--limit N] [--n-boot N]

Reads the frozen panel and the published three-state run; writes only into
--out. Nothing under outputs/ or experiments/<older run>/ is touched.
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

# One BLAS thread per process: the fold loop is already parallel over patients,
# so letting each worker spawn a thread pool oversubscribes the node. Set
# before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import loo, next_visit as nv  # noqa: E402
from ppmi_subtype import two_state_hmm as ts  # noqa: E402
from ppmi_subtype import two_state_plots as tp  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

#: The frozen visit panel, as scripts/60 and scripts/87 check it.
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

#: The published next-visit run. Read only, to prove the 3x3 arm refitted here
#: reproduces the existing model rather than merely resembling it.
PUBLISHED_RUN = ROOT / "experiments/20261020T104756Z_next_visit_td_pigd_prediction"

#: Starts for the full-cohort reporting fits. More than the fold loop uses,
#: and genuinely dispersed, because this is where a local optimum would be
#: reported if there were one.
REPORT_STARTS = 16

lines: list[str] = []
_WORK: dict = {}


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def section(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def block(frame, **kw) -> str:
    return "    " + frame.to_string(**kw).replace("\n", "\n    ")


def _slug(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ---------------------------------------------------------------------------
#  Fold workers
# ---------------------------------------------------------------------------
def _init(data, targets, seed, n_restarts, emission_floor) -> None:
    _WORK.update(data=data, targets=targets, seed=seed, n_restarts=n_restarts,
                 emission_floor=emission_floor)


def _fold(patno):
    try:
        return ts.run_fold(_WORK["data"], _WORK["targets"], patno, _WORK["seed"],
                           _WORK["n_restarts"], _WORK["emission_floor"]), None
    except Exception as exc:                        # noqa: BLE001 -- recorded, not hidden
        return None, f"{type(exc).__name__}: {exc}"


def run_folds(data, targets, patients, seed, n_restarts, emission_floor, workers):
    frames, failures = [], []
    if workers <= 1:
        for n, patno in enumerate(patients, start=1):
            frame, error = (ts.run_fold(data, targets, patno, seed, n_restarts,
                                        emission_floor), None)
            frames.append(frame)
            if n % 50 == 0:
                say(f"  fold {n:,}/{len(patients):,}")
    else:
        args = (data, targets, seed, n_restarts, emission_floor)
        with mp.get_context("fork").Pool(workers, initializer=_init,
                                         initargs=args) as pool:
            for n, (frame, error) in enumerate(
                    pool.imap(_fold, patients, chunksize=4), start=1):
                if error is None:
                    frames.append(frame)
                else:
                    failures.append({"held_out_patient": patients[n - 1],
                                     "error": error})
                if n % 100 == 0:
                    say(f"  fold {n:,}/{len(patients):,}  "
                        f"({time.time() - _WORK.get('t0', time.time()):.0f}s)")

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return predictions, pd.DataFrame(failures, columns=["held_out_patient", "error"])


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N folds (smoke test)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    cfg = load_config()
    started = time.time()
    _WORK["t0"] = started

    # ------------------------------------------------------------------ data
    section("1. INPUT")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    say(f"  panel                {panel_path.relative_to(ROOT)}")
    say(f"  md5                  {panel_md5}")
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}")
    say(f"  expected md5         {EXPECTED_PANEL_MD5}  [MATCH]")

    states = [cfg.subtype.labels[s] for s in ts.THREE_LABELS]
    panel = pd.read_csv(panel_path, low_memory=False)
    base = loo.prepare(panel, states)
    data = ts.prepare(base)
    targets = ts.build_targets(data)
    counts = data.counts()

    say(f"  labels               {dict(zip(ts.THREE_LABELS, states))}")
    say(f"  emission floor       {cfg.model['emission_floor']}  (square arms only)")
    say(f"  restarts per fold    {cfg.model['n_restarts']}")
    say(f"  starts, report fits  {REPORT_STARTS} (dispersed)")
    say(f"  seed                 {cfg.seed}")
    say(f"  bootstrap draws      {args.n_boot}")

    # ------------------------------------------------------------- the cohort
    section("2. DATASET AND THE TWO OBSERVATION VIEWS")
    say("  Both models draw on the same cohort and the same frozen panel. The")
    say("  only difference is which visits enter the observation sequences.")
    say("")
    cohort = pd.DataFrame([
        {"view": "Model 2 (TD / Indeterminate / PIGD)",
         "patients": counts["model2_patients"], "visits": counts["model2_visits"],
         "transitions": counts["model2_transitions"]},
        {"view": "Model 1 (TD / PIGD, Indeterminate excluded)",
         "patients": counts["model1_patients"], "visits": counts["model1_visits"],
         "transitions": counts["model1_transitions"]},
    ])
    say(block(cohort, index=False))
    cohort.to_csv(out / "cohort_counts.csv", index=False)
    say("")
    say(f"  Indeterminate visits dropped for Model 1      "
        f"{counts['indeterminate_visits_dropped']:,}")
    say(f"  visits excluded from Model 1 in total         "
        f"{counts['visits_excluded_from_model1']:,}")
    say(f"  patients excluded from Model 1                "
        f"{counts['patients_excluded_from_model1']:,}  "
        f"(fewer than two TD/PIGD visits, so no transition to observe)")
    say("")
    say("  The two numbers differ because dropping an Indeterminate visit can")
    say("  leave a patient with fewer than two retained visits, which removes")
    say("  their remaining visits from the Model 1 sequences as well.")
    say("")
    elig = targets[targets["model1_eligible"]]
    longer = int((elig["model1_interval_years"]
                  > elig["interval_years"] + 1e-9).sum())
    say("  Elapsed time between retained visits is the ACTUAL gap, not a count")
    say("  of annual visits. Where an Indeterminate visit was dropped between")
    say("  two retained ones, Model 1 sees the wider gap:")
    say(f"    eligible targets whose Model 1 gap is wider   {longer:,} / {len(elig):,}")
    say(f"    median gap, three-label models                "
        f"{elig['interval_years'].median():.3f} y")
    say(f"    median gap, Model 1                           "
        f"{elig['model1_interval_years'].median():.3f} y")
    say(f"    max gap, Model 1                              "
        f"{elig['model1_interval_years'].max():.3f} y")

    say("")
    say("  Actual label at the target visit, over all prediction targets:")
    lab = targets["actual_label"].value_counts().reindex(ts.THREE_LABELS)
    say(block(pd.DataFrame({"n": lab, "share": (lab / len(targets)).round(4)})))
    say("")
    say(f"  prediction targets, three-label models   {len(targets):,}")
    say(f"  prediction targets, Model 1 (= common)   "
        f"{int(targets['model1_eligible'].sum()):,}")
    say(f"  TD/PIGD targets Model 1 cannot predict   "
        f"{int((~targets['model1_eligible'] & (targets['target_index'] != ts.IND)).sum()):,}"
        f"  (no TD/PIGD visit precedes them)")
    targets.to_csv(out / "prediction_targets.csv", index=False)

    # ------------------------------------------------- full-cohort reporting fits
    section("3. PARAMETERS, FITTED ON THE WHOLE LABELLED COHORT")
    say("  These fits exist to report parameters and information criteria, which")
    say("  AIC and BIC require at the model's own maximum. They are NOT used for")
    say("  any held-out number: every prediction in sections 5 onward comes from")
    say("  a fold that never saw the patient it predicts.")
    say("")
    say(f"  {REPORT_STARTS} dispersed starts per model (hmm_stability.dispersed_starts):")
    say("  start 0 is the project default, the rest are spread over the parameter")
    say("  space. The best is selected on the TRAINING log-likelihood.")

    fits = ts.fit_all(data, cfg.model["emission_floor"], cfg.seed, REPORT_STARTS,
                      dispersed=True)

    spread_rows, crit_rows = [], []
    for model, vfit in fits.items():
        say("")
        say(f"  --- {model}")
        say(f"      hidden states {vfit.n_hidden}   observed labels {vfit.n_observed}"
            f"   free parameters {vfit.n_params}")
        say(f"      emission floor {vfit.emission_floor}")
        say(f"      training log-likelihood {vfit.loglik:,.4f} over "
            f"{vfit.n_observations:,} visits")
        say(f"      converged {vfit.converged}   "
            f"projected gradient {vfit.best.proj_grad:.3e}")
        say("      generator Q (per year):")
        say(block(vfit.generator_frame().round(5)))
        say("      emission E = P(observed label | hidden state):")
        say(block(vfit.emission_frame().round(5)))
        say("      initial distribution:")
        say(block(vfit.initial_frame().round(5).to_frame()))
        say("      mean sojourn (years):")
        say(block(vfit.sojourn_frame().round(4).to_frame()))

        slug = _slug(model)
        vfit.generator_frame().to_csv(out / f"params_Q_{slug}.csv")
        vfit.emission_frame().to_csv(out / f"params_E_{slug}.csv")
        vfit.initial_frame().to_frame().to_csv(out / f"params_pi_{slug}.csv")

        spread_rows.append({"model": model, **vfit.spread()})
        crit_rows.append({"model": model, "n_hidden": vfit.n_hidden,
                          "n_observed": vfit.n_observed,
                          "emission_floor": vfit.emission_floor,
                          **vfit.criteria()})

    spread = pd.DataFrame(spread_rows)
    spread.to_csv(out / "multistart_spread.csv", index=False)
    say("")
    say("  Multi-start behaviour (how much the optimum depends on where it starts):")
    disp_s = spread.copy()
    disp_s["model"] = disp_s["model"].map(tp.short)
    say(block(disp_s[["model", "n_starts", "n_at_best_loglik", "share_at_best",
                      "n_distinct_loglik_levels", "max_dQ_at_best_level",
                      "max_dE_at_best_level", "best_loglik",
                      "worst_start_loglik"]].round(4), index=False))
    say("")
    say("  share_at_best is the fraction of starts landing within 0.01 log-")
    say("  likelihood of the best. max_dQ / max_dE is the largest difference in")
    say("  any canonical generator or emission entry AMONG those starts: a large")
    say("  value there means a flat ridge --- the same likelihood reached at")
    say("  different parameters --- which is the failure mode that matters for an")
    say("  unconstrained two-state model, not a merely lower optimum elsewhere.")

    # ----------------------------------------------------- structural checks
    section("4. STRUCTURAL CHECKS ON THE FITTED PARAMETERS")
    pchecks = ts.parameter_checks(fits)
    for name, ok in sorted(pchecks.items()):
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "parameter_checks.json").write_text(json.dumps(pchecks, indent=2),
                                               encoding="utf-8")
    if not all(pchecks.values()):
        raise SystemExit(f"parameter checks failed: "
                         f"{[k for k, v in pchecks.items() if not v]}")

    tp.emission_heatmaps(fits, out / "figures/emission_matrices.png",
                         order=[ts.MODEL_2X2, ts.MODEL_2X3, ts.MODEL_3X3])

    # ------------------------------------------------------------------ LOPO
    section("5. LEAVE-ONE-PATIENT-OUT PREDICTION")
    patients = data.patients[: args.limit] if args.limit else data.patients
    say(f"  folds                {len(patients):,}")
    say(f"  workers              {args.workers}")
    say("  Each fold refits all four CT-HMM arms on the other patients only, in")
    say("  both observation views, and recomputes the rule's and the null's")
    say("  priors from the training patients. The held-out patient is whole on")
    say("  one side: loo.assert_no_leakage is called on both views every fold.")
    say("")
    say("  Prediction is forward filtering only. alpha is advanced through the")
    say("  labels at the visits BEFORE the target and propagated through")
    say("  exp(Q dt); no backward pass runs and nothing depending on the target")
    say("  label enters. Elapsed time to the target is an input, as throughout")
    say("  this project's panel models: the model is told when the next")
    say("  examination happens, never what it records.")

    predictions, failures = run_folds(
        data, targets, patients, cfg.seed, cfg.model["n_restarts"],
        cfg.model["emission_floor"], args.workers)
    if not len(predictions):
        raise SystemExit("every fold failed; nothing to evaluate")

    predictions = predictions.sort_values(
        ["model", "PATNO", "visit_index"]).reset_index(drop=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    failures.to_csv(out / "failed_folds.csv", index=False)

    say("")
    say(f"  folds attempted      {len(patients):,}")
    say(f"  folds completed      {predictions['held_out_patient'].nunique():,}")
    say(f"  folds failed         {len(failures):,}")
    if len(failures):
        say(block(failures, index=False))
    say(f"  prediction rows      {len(predictions):,}")
    say(f"  elapsed              {time.time() - started:.0f}s")

    conv_rows = []
    for model, g in predictions.groupby("model", sort=False):
        folds = g.drop_duplicates("held_out_patient")
        conv_rows.append({
            "model": model, "n_folds": int(len(folds)),
            "n_scipy_success": int(folds["fit_scipy_success"].sum()),
            "n_converged": int(folds["fit_converged"].sum()),
            "n_not_converged": int((~folds["fit_converged"].astype(bool)).sum()),
            "max_proj_grad": float(np.nanmax(folds["fit_proj_grad"]))
            if folds["fit_proj_grad"].notna().any() else float("nan"),
            "mean_train_loglik": float(folds["fit_train_loglik"].mean()),
        })
    convergence = pd.DataFrame(conv_rows)
    convergence.to_csv(out / "convergence.csv", index=False)
    say("")
    say("  Convergence across folds (a fold counts as converged when scipy")
    say(f"  succeeds OR the projected gradient is below {ts.PROJ_GRAD_TOL}; see")
    say("  src/ppmi_subtype/hmm_stability.py for why the flag alone is not it):")
    say(block(convergence.round(4), index=False))

    # ---------------------------------------------------------------- checks
    section("6. LEAKAGE AND CORRECTNESS CHECKS")
    checks = ts.prediction_checks(predictions, targets, data, failures, patients)
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "prediction_checks.json").write_text(json.dumps(checks, indent=2),
                                                encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # ----------------------------------------------- comparison A: native
    section("7. COMPARISON A: NATIVE PREDICTION, EACH MODEL IN ITS OWN SPACE")
    say("  The two tasks are NOT comparable and are never averaged. A three-class")
    say("  prediction is harder than a two-class one on its face, and the sets of")
    say("  eligible target visits differ:")
    say("")
    say(f"    three-class task   {len(targets):,} targets over "
        f"{targets['PATNO'].nunique():,} patients, labels TD / Indeterminate / PIGD")
    say(f"    two-class task     {int(targets['model1_eligible'].sum()):,} targets over "
        f"{elig['PATNO'].nunique():,} patients, labels TD / PIGD")
    say("")
    say("  Model 1 is absent from the three-class table because it has no")
    say("  Indeterminate category to predict. Scoring it there would mean giving")
    say("  it a zero for a class it was never asked about.")

    three_models = [m for m in (ts.MODEL_2X3, ts.MODEL_3X3, ts.MODEL_PERSIST,
                                ts.MODEL_MARGINAL)]
    three_patients = sorted(set(ts.scored(
        predictions[predictions["model"] == ts.MODEL_2X3], ts.TASK_THREE)["PATNO"]))
    draws_three = ts.patient_draws(three_patients, args.n_boot, cfg.seed)

    native_three = pd.DataFrame([
        ts.evaluate(predictions[predictions["model"] == m], ts.TASK_THREE,
                    draws=draws_three, patients=three_patients)
        for m in three_models]).sort_values("accuracy", ascending=False)
    native_three.to_csv(out / "native_three_class.csv", index=False)

    show3 = ["model", "n_scored", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
             "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
             "balanced_accuracy", "log_loss", "n_ties"]
    say("")
    say("  Native THREE-CLASS prediction (TD / Indeterminate / PIGD):")
    say(block(native_three[show3].round(4), index=False))

    two_patients = sorted(set(ts.scored(
        predictions[predictions["model"] == ts.MODEL_2X2], ts.TASK_TWO)["PATNO"]))
    draws_two = ts.patient_draws(two_patients, args.n_boot, cfg.seed)

    native_two = pd.DataFrame([
        ts.evaluate(predictions[predictions["model"] == m], ts.TASK_TWO,
                    draws=draws_two, patients=two_patients)
        for m in (ts.MODEL_2X2, ts.MODEL_2X2_FREE)])
    native_two.to_csv(out / "native_two_class_model1.csv", index=False)
    say("")
    say("  Native TWO-CLASS prediction, Model 1 on its own cases:")
    say(block(native_two[show3].round(4), index=False))

    for m, g in predictions.groupby("model", sort=False):
        slug = _slug(m)
        if m not in ts.TWO_LABEL_MODELS:
            ts.confusion_frame(g, ts.TASK_THREE).to_csv(
                out / f"confusion_three_class_{slug}.csv")
            ts.per_class_frame(g, ts.TASK_THREE).round(4).to_csv(
                out / f"per_class_three_class_{slug}.csv", index=False)

    say("")
    say(f"  Confusion matrix, {ts.MODEL_2X3} (rows actual, columns predicted):")
    say(block(ts.confusion_frame(
        predictions[predictions["model"] == ts.MODEL_2X3], ts.TASK_THREE)))
    say("")
    say("  Per class:")
    say(block(ts.per_class_frame(
        predictions[predictions["model"] == ts.MODEL_2X3],
        ts.TASK_THREE).round(4), index=False))
    say("")
    say(f"  Confusion matrix, {ts.MODEL_3X3} (rows actual, columns predicted):")
    say(block(ts.confusion_frame(
        predictions[predictions["model"] == ts.MODEL_3X3], ts.TASK_THREE)))
    say("")
    say("  Per class:")
    say(block(ts.per_class_frame(
        predictions[predictions["model"] == ts.MODEL_3X3],
        ts.TASK_THREE).round(4), index=False))

    # --------------------------------------------- comparison B: common set
    section("8. COMPARISON B: THE COMMON TD/PIGD EVALUATION")
    say("  Every arm, scored on the IDENTICAL target visits: the ones Model 1 can")
    say("  predict, where the actual next label is TD or PIGD and a TD or PIGD")
    say("  visit precedes it. For the three-label models the distribution is")
    say("  renormalised onto {TD, PIGD} by next_visit.conditional_two_class:")
    say("")
    say("      P(TD | TD or PIGD) = p_TD / (p_TD + p_PIGD)")
    say("")
    say(f"  shared transitions   {len(two_patients) and int(targets['model1_eligible'].sum()):,}")
    say(f"  shared patients      {len(two_patients):,}")
    say(f"  tie rule             {nv.TIE_RULE} (two-class), ties counted below")

    common = pd.DataFrame([
        ts.evaluate(predictions[predictions["model"] == m], ts.TASK_TWO,
                    draws=draws_two, patients=two_patients)
        for m in ts.ALL_MODELS]).sort_values("accuracy", ascending=False)
    common.to_csv(out / "common_two_class.csv", index=False)
    say("")
    say(block(common[show3 + ["brier", "rule_based_probabilities"]].round(4),
              index=False))

    scored_sets = {m: frozenset(map(tuple, ts.scored(g, ts.TASK_TWO)[
        ["PATNO", "visit_index"]].to_numpy()))
        for m, g in predictions.groupby("model", sort=False)}
    identical = len(set(scored_sets.values())) == 1
    say("")
    say(f"  scored-set sizes     {dict((tp.short(k), len(v)) for k, v in scored_sets.items())}")
    say(f"  all arms scored the identical set of transitions: {identical}")
    if not identical:
        raise SystemExit("the common comparison is not on identical transitions")

    for m, g in predictions.groupby("model", sort=False):
        slug = _slug(m)
        ts.confusion_frame(g, ts.TASK_TWO).to_csv(out / f"confusion_common_{slug}.csv")
        ts.per_class_frame(g, ts.TASK_TWO).round(4).to_csv(
            out / f"per_class_common_{slug}.csv", index=False)
        ts.calibration(g).round(4).to_csv(out / f"calibration_common_{slug}.csv",
                                          index=False)

    for m in (ts.MODEL_2X2, ts.MODEL_2X3, ts.MODEL_3X3, ts.MODEL_PERSIST):
        g = predictions[predictions["model"] == m]
        say("")
        say(f"  Confusion matrix on the common subset, {m}:")
        say(block(ts.confusion_frame(g, ts.TASK_TWO)))
        say(block(ts.per_class_frame(g, ts.TASK_TWO).round(4), index=False))

    tp.accuracy_intervals(
        common, out / "figures/common_accuracy.png",
        "Common TD/PIGD evaluation: identical held-out transitions",
        f"{int(common['n_scored'].max()):,} transitions, "
        f"{int(common['n_patients'].max()):,} patients, "
        f"95% patient bootstrap ({args.n_boot:,} draws)")
    tp.accuracy_intervals(
        native_three, out / "figures/native_three_class_accuracy.png",
        "Native three-class evaluation (TD / Indeterminate / PIGD)",
        f"{int(native_three['n_scored'].max()):,} transitions, "
        f"{int(native_three['n_patients'].max()):,} patients")
    tp.calibration_curves(
        {m: ts.calibration(predictions[predictions["model"] == m])
         for m in (ts.MODEL_2X2, ts.MODEL_2X3, ts.MODEL_3X3)},
        out / "figures/calibration_common.png",
        "Calibration of P(PIGD) on the common subset")

    # ------------------------------------------------------- paired contrasts
    section("9. PAIRED DIFFERENCES ON THE COMMON SUBSET")
    say("  Each row is accuracy_a - accuracy_b on the transitions both arms")
    say("  scored, with a 95% interval from resampling PATIENTS. Comparing the")
    say("  two separate intervals in section 8 would be the wrong test: the same")
    say("  transitions score both arms, so the comparison has to be paired. All")
    say("  contrasts and all single-model intervals use the SAME patient draws.")

    pairs = [
        (ts.MODEL_2X3, ts.MODEL_2X2),
        (ts.MODEL_2X2, ts.MODEL_3X3),
        (ts.MODEL_2X3, ts.MODEL_3X3),
        (ts.MODEL_2X2, ts.MODEL_2X2_FREE),
        (ts.MODEL_2X2, ts.MODEL_PERSIST),
        (ts.MODEL_2X3, ts.MODEL_PERSIST),
        (ts.MODEL_3X3, ts.MODEL_PERSIST),
        (ts.MODEL_2X2, ts.MODEL_MARGINAL),
    ]
    contrasts = pd.DataFrame([
        ts.paired_difference(predictions[predictions["model"] == a],
                             predictions[predictions["model"] == b],
                             ts.TASK_TWO, draws_two, two_patients)
        for a, b in pairs])
    contrasts.to_csv(out / "paired_contrasts_common.csv", index=False)
    showc = ["model_a", "model_b", "accuracy_a", "accuracy_b", "accuracy_delta",
             "delta_ci_low", "delta_ci_high", "ci_includes_zero",
             "a_right_b_wrong", "b_right_a_wrong",
             "mcnemar_exact_p_ignores_clustering"]
    disp = contrasts.copy()
    disp["model_a"] = disp["model_a"].map(tp.short)
    disp["model_b"] = disp["model_b"].map(tp.short)
    say("")
    say(block(disp[showc].round(4), index=False))

    say("")
    say("  Paired differences on the native THREE-CLASS task (Model 2 vs the")
    say("  existing three-state model, same targets, same observation space):")
    three_contrasts = pd.DataFrame([
        ts.paired_difference(predictions[predictions["model"] == a],
                             predictions[predictions["model"] == b],
                             ts.TASK_THREE, draws_three, three_patients)
        for a, b in [(ts.MODEL_2X3, ts.MODEL_3X3),
                     (ts.MODEL_2X3, ts.MODEL_PERSIST),
                     (ts.MODEL_3X3, ts.MODEL_PERSIST)]])
    three_contrasts.to_csv(out / "paired_contrasts_three_class.csv", index=False)
    disp3 = three_contrasts.copy()
    disp3["model_a"] = disp3["model_a"].map(tp.short)
    disp3["model_b"] = disp3["model_b"].map(tp.short)
    say(block(disp3[showc].round(4), index=False))

    # ------------------------------------------------------- model selection
    section("10. MODEL SELECTION: AIC AND BIC")
    criteria = pd.DataFrame(crit_rows)
    criteria["observation_set"] = np.where(
        criteria["model"].isin(ts.TWO_LABEL_MODELS),
        "TD/PIGD visits only", "all labelled visits")
    criteria.to_csv(out / "aic_bic.csv", index=False)
    say("  n in BIC is the number of likelihood terms: one per VISIT in the")
    say("  sequences that were fitted, taken from hmm.HMMFit.n_observations.")
    say("  That is this project's convention, used by")
    say("  scripts/65_model_selection_criteria.py. The HMM's likelihood")
    say("  factorises over whole sequences and every visit contributes a term,")
    say("  including each patient's first, which the initial distribution")
    say("  accounts for; the transition count would undercount it by exactly")
    say("  the number of patients.")
    say("")
    say(block(criteria[["model", "n_hidden", "n_observed", "observation_set",
                        "loglik", "n_params", "n_observations", "aic", "bic"]]
              .round(3), index=False))
    say("")
    say("  COMPARABLE: Model 2 against the existing three-state model. Both are")
    say("  likelihoods over the same 4,295 visits and the same three-label")
    say("  alphabet, so the difference in AIC and BIC is a difference in fit per")
    say("  parameter on one set of observations.")
    comp = criteria.set_index("model")
    if {ts.MODEL_2X3, ts.MODEL_3X3} <= set(comp.index):
        pair = comp.loc[[ts.MODEL_2X3, ts.MODEL_3X3]]
        say(block((pair[["loglik", "n_params", "aic", "bic"]]
                   - pair[["loglik", "n_params", "aic", "bic"]].min()).round(3)))
        say(f"    delta AIC (M2 - existing)  "
            f"{comp.loc[ts.MODEL_2X3, 'aic'] - comp.loc[ts.MODEL_3X3, 'aic']:+.3f}")
        say(f"    delta BIC (M2 - existing)  "
            f"{comp.loc[ts.MODEL_2X3, 'bic'] - comp.loc[ts.MODEL_3X3, 'bic']:+.3f}")
    say("")
    say("  NOT COMPARABLE: Model 1 against either of them. Its likelihood is")
    say("  over 3,812 visits, not 4,295, and over a two-label alphabet. Its AIC")
    say("  and BIC are reported for completeness and for comparison with the")
    say("  no-floor variant of itself, which shares its observations exactly.")
    say("  A lower AIC there reflects fewer observations, not a better model.")
    say("")
    say("  AIC and BIC also assume independent observations. Visits are")
    say("  clustered within patients, so the effective sample size is below the")
    say("  term count and BIC's penalty is, if anything, too weak here --- the")
    say("  same caution docs/model_comparison/01 records.")

    # ------------------------------------------- reproducing the existing model
    section("11. THE EXISTING THREE-STATE RESULTS ARE UNCHANGED AND REPRODUCED")
    repro: dict = {"published_run": str(PUBLISHED_RUN.relative_to(ROOT))}
    manifest = PUBLISHED_RUN / "MANIFEST.sha256"
    if manifest.exists():
        bad = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            path = PUBLISHED_RUN / rel
            if not path.exists() or sha(path) != digest:
                bad.append(rel)
        repro["published_manifest_verified"] = not bad
        repro["published_files_changed"] = bad
        say(f"  published MANIFEST.sha256 verifies: {not bad}")
        if bad:
            say(f"    files that do not match: {bad}")

        pub = pd.read_csv(PUBLISHED_RUN / "next_visit_predictions.csv",
                          low_memory=False)
        pub = pub[pub["model"] == nv.MODEL_CTHMM].copy()
        # The published table indexes a case by the visit it predicts FROM;
        # this run indexes it by the visit it predicts. They differ by one.
        pub["visit_index"] = pub["visit_index"] + 1
        mine = predictions[predictions["model"] == ts.MODEL_3X3]
        both = pub.merge(mine, on=["PATNO", "visit_index"], suffixes=("_pub", "_new"))
        d_td = (both["p_TD_raw"] - both["p_TD"]).abs()
        d_pigd = (both["p_PIGD_raw"] - both["p_PIGD"]).abs()
        d_ind = (both["p_Indeterminate_raw"] - both["p_Indeterminate"]).abs()
        worst = float(max(d_td.max(), d_pigd.max(), d_ind.max()))
        repro.update({
            "n_cases_matched": int(len(both)),
            "max_abs_probability_difference": worst,
            "reproduces_published_cthmm": bool(worst < 1e-8),
        })
        say(f"  cases matched against the published CT-HMM   {len(both):,}")
        say(f"  largest absolute difference in any of the three probabilities "
            f"{worst:.3e}")
        say(f"  the 3x3 arm reproduces the published model: {worst < 1e-8}")
        say("")
        say("  This is the check that the existing model is untouched. The 3x3")
        say("  arm is refitted here by the same code from the same panel; if its")
        say("  per-case probabilities equal the published ones, then the")
        say("  comparison is against the existing model itself and not a")
        say("  look-alike, and the published run's own files are byte-identical.")

        pub_acc = pd.read_csv(PUBLISHED_RUN / "model_comparison.csv")
        pub_cthmm = pub_acc[pub_acc["model"] == nv.MODEL_CTHMM].iloc[0]
        repro["published_accuracy_all_eligible"] = float(pub_cthmm["accuracy"])
        repro["published_n_scored"] = int(pub_cthmm["n_scored"])
        say("")
        say(f"  For the record, the published headline is accuracy "
            f"{pub_cthmm['accuracy']:.4f} over {int(pub_cthmm['n_scored']):,}")
        say("  eligible transitions. That is a LARGER set than this experiment's")
        say("  common subset: it includes TD/PIGD targets with no preceding")
        say("  TD/PIGD visit, which Model 1 cannot predict at all. The common")
        say("  number in section 8 is therefore not the published number, and")
        say("  the two must not be quoted against each other.")
    else:
        repro["published_manifest_verified"] = None
        say(f"  no MANIFEST at {manifest}; skipped")
    (out / "reproduction_check.json").write_text(
        json.dumps(repro, indent=2, default=float), encoding="utf-8")

    # ---------------------------------------------------------- breakdowns
    section("12. WHERE THE DIFFERENCES LIVE (COMMON SUBSET)")
    breakdowns = ts.accuracy_breakdowns(predictions, ts.TASK_TWO)
    for key, frame in breakdowns.items():
        frame.round(4).to_csv(out / f"accuracy_common_{key}.csv", index=False)
        disp_b = frame.copy()
        disp_b["model"] = disp_b["model"].map(tp.short)
        say("")
        say(f"  {key}:")
        say(block(disp_b.round(4), index=False))

    # --------------------------------------------------------- provenance
    section("13. PROVENANCE")
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "node": platform.node(),
        "workers": args.workers,
        "seed": cfg.seed,
        "n_restarts_per_fold": cfg.model["n_restarts"],
        "report_starts": REPORT_STARTS,
        "n_boot": args.n_boot,
        "emission_floor": cfg.model["emission_floor"],
        "panel_md5": panel_md5,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    env["code_sha256"] = {
        str(p.relative_to(ROOT)): sha(p) for p in
        [Path(__file__), ROOT / "src/ppmi_subtype/two_state_hmm.py",
         ROOT / "src/ppmi_subtype/two_state_plots.py",
         ROOT / "src/ppmi_subtype/hmm.py",
         ROOT / "src/ppmi_subtype/hmm_stability.py",
         ROOT / "src/ppmi_subtype/ctmc.py", ROOT / "src/ppmi_subtype/loo.py",
         ROOT / "src/ppmi_subtype/next_visit.py"]
    }
    for k, v in env.items():
        if k != "code_sha256":
            say(f"  {k:<24} {v}")

    summary = {
        "environment": env,
        "design": {
            "task": "next observed motor subtype at a held-out patient's visit",
            "validation": "leave-one-patient-out; the whole patient is held out "
                          "of every fit in both observation views and out of the "
                          "rule's and the null's priors",
            "filtering": "forward recursion only; no backward pass and no "
                          "quantity depending on the target label",
            "elapsed_time": "dt to the target visit is an input to every arm",
            "tie_rule": {"two_class": nv.TIE_RULE,
                         "three_class": [ts.THREE_LABELS[k] for k in ts.TIE_ORDER]},
            "model1_indeterminate": "excluded from the sequences; the actual "
                                    "elapsed time between retained visits is kept",
            "model2_indeterminate": "an observed label either hidden state may "
                                    "emit, with the probability learned",
            "emission_floor": "applies to the square arms only; undefined for "
                              "2 hidden x 3 observed, so Model 2 is unconstrained "
                              "and a no-floor 2x2 arm is fitted as the control",
        },
        "counts": {**counts, "n_targets": len(targets),
                   "n_common_transitions": int(targets["model1_eligible"].sum()),
                   "n_common_patients": len(two_patients),
                   "n_folds_attempted": len(patients),
                   "n_folds_failed": len(failures),
                   "target_label_counts": lab.to_dict()},
        "parameter_checks": pchecks,
        "prediction_checks": checks,
        "multistart_spread": spread.to_dict(orient="records"),
        "parameters": {
            m: {"Q": v.generator_frame().to_dict(),
                "E": v.emission_frame().to_dict(),
                "pi": v.initial_frame().to_dict(),
                "n_params": v.n_params, "loglik": v.loglik,
                "n_observations": v.n_observations,
                "converged": v.converged}
            for m, v in fits.items()},
        "aic_bic": criteria.to_dict(orient="records"),
        "native_three_class": native_three.to_dict(orient="records"),
        "native_two_class_model1": native_two.to_dict(orient="records"),
        "common_two_class": common.to_dict(orient="records"),
        "paired_contrasts_common": contrasts.to_dict(orient="records"),
        "paired_contrasts_three_class": three_contrasts.to_dict(orient="records"),
        "convergence": convergence.to_dict(orient="records"),
        "reproduction_check": repro,
        "accuracy_breakdowns": {k: v.to_dict(orient="records")
                                for k, v in breakdowns.items()},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),
        encoding="utf-8")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
