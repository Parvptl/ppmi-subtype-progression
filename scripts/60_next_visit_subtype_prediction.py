"""Next-visit TD-vs-PIGD prediction from a patient's TD/Indeterminate/PIGD history.

For every labelled visit that is followed by another labelled visit, seven
candidate models produce two conditional probabilities,

    P_TD   = P(next label is TD   | next label is TD or PIGD, history to date)
    P_PIGD = P(next label is PIGD | next label is TD or PIGD, history to date)

under leave-one-patient-out validation: the patient is held out whole, every
model and every piece of learned preprocessing is fitted on the other 933, and
the prediction for each of the held-out patient's visits uses only that
patient's earlier labels plus the date of the next examination.

The candidates span the two axes the scientific question asks about -- latent
state and elapsed time -- so the comparison can attribute any difference:

    CT-HMM (latent, interval-aware)       latent CTMC + emission matrix
    CTMC (observed, interval-aware)       no latent layer, interval-aware
    Markov (observed, discrete)           no latent layer, interval-agnostic
    Direct logistic (history + interval)  discriminative, interval-aware
    Direct logistic (history only)        discriminative, interval-agnostic
    Persistence (last non-Indeterminate)  a rule
    Marginal (two-class prior)            the null

Indeterminate stays a third observed label in every input history and is never
recoded. Visits whose *actual* next label is Indeterminate are reported
separately and never scored as a hit or a miss. See the module docstring of
`ppmi_subtype.next_visit` for what the two probabilities do and do not mean.

    python scripts/60_next_visit_subtype_prediction.py --out <dir> [--workers N]
                                                       [--limit N]

Writes the prediction table, per-model metrics, confusion matrices, calibration
and a MANIFEST into --out. Reads the frozen panel; writes nothing under
outputs/.
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

# One BLAS thread per process. The fold loop is already parallel over
# patients, so letting each worker spawn a full thread pool oversubscribes
# the node and makes every fold slower. Set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import loo, next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

#: The frozen visit panel. Checked on load, so a prediction table
#: can never be attributed to the wrong input.
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

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
    """A filename for a model name, distinct for every candidate.

    The two direct-logistic variants differ only inside their parentheses, so a
    slug built from the part before the bracket collides and one file silently
    overwrites the other. The whole name is kept.
    """
    out = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ---------------------------------------------------------------------------
#  Fold workers
# ---------------------------------------------------------------------------
def _init(data, cases, seed, n_restarts, emission_floor) -> None:
    _WORK.update(data=data, cases=cases, seed=seed, n_restarts=n_restarts,
                 emission_floor=emission_floor)


def _fold(patno):
    """Run one fold; return (frame, None) or (None, error string)."""
    try:
        return nv.run_fold(_WORK["data"], _WORK["cases"], patno, _WORK["seed"],
                           _WORK["n_restarts"], _WORK["emission_floor"]), None
    except Exception as exc:                    # noqa: BLE001 -- recorded, not hidden
        return None, f"{type(exc).__name__}: {exc}"


def run_folds(data, cases, patients, seed, n_restarts, emission_floor, workers):
    """The LOPO loop, optionally across processes. Failures are collected."""
    if workers <= 1:
        return nv.run_lopo(data, cases, seed, n_restarts, emission_floor,
                           patients=patients)

    frames, failures = [], []
    args = (data, cases, seed, n_restarts, emission_floor)
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=args) as pool:
        for n, (frame, error) in enumerate(pool.imap(_fold, patients, chunksize=4),
                                           start=1):
            if error is None:
                frames.append(frame)
            else:
                failures.append({"held_out_patient": patients[n - 1], "error": error})
            if n % 100 == 0:
                say(f"  fold {n:,}/{len(patients):,}")

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return predictions, pd.DataFrame(failures,
                                     columns=["held_out_patient", "error"])


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory to write into")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the fold loop (folds are independent)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N folds (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    # ------------------------------------------------------------------ data
    section("1. INPUT")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    say(f"  panel                {panel_path.relative_to(ROOT)}")
    say(f"  md5                  {panel_md5}")
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}")
    say(f"  expected md5         {EXPECTED_PANEL_MD5}  [MATCH]")

    states = [cfg.subtype.labels[s] for s in nv.STATE_NAMES]
    panel = pd.read_csv(panel_path, low_memory=False)
    data = loo.prepare(panel, states)
    cases = nv.build_cases(data)

    say(f"  labels               {dict(zip(nv.STATE_NAMES, states))}")
    say(f"  patients with >= 2 labelled visits   {len(data.sequences):,}")
    say(f"  prediction cases (transitions)       {len(cases):,}")
    say(f"  emission floor       {cfg.model['emission_floor']}")
    say(f"  restarts per fit     {cfg.model['n_restarts']}")
    say(f"  seed                 {cfg.seed}")

    section("2. WHAT IS PREDICTED, AND WHAT THE TWO NUMBERS MEAN")
    say("  P_TD and P_PIGD are CONDITIONAL on the next observed label being TD")
    say("  or PIGD. They are not unconditional next-visit probabilities: the")
    say("  generative models place mass on Indeterminate too, and that mass is")
    say("  divided out by the renormalisation")
    say("")
    say("      P_TD = p_TD / (p_TD + p_PIGD),  P_PIGD = p_PIGD / (p_TD + p_PIGD)")
    say("")
    say("  p_Indeterminate is kept in the prediction table as")
    say("  p_Indeterminate_raw and reported, not discarded. A visit whose actual")
    say("  next label is Indeterminate is a case where the conditioning event")
    say("  did not occur, so it is counted separately and never scored.")
    say("")
    say(f"  Tie rule: P_TD == P_PIGD resolves to {nv.TIE_RULE}, because a")
    say("  symmetric probability carries no information and the error that")
    say("  misses a PIGD conversion is the costlier one. Ties are counted.")
    say("")
    say("  Elapsed time to the next visit is an INPUT to the interval-aware")
    say("  models. They are told when the next examination happens, never what")
    say("  it records. That is this project's panel-data convention and it is an")
    say("  assumption, because PPMI's visit schedule is not independent of")
    say("  disease course.")

    # ---------------------------------------------------------- label counts
    section("3. OUTCOME COUNTS")
    counts = cases["actual_next_label"].value_counts().reindex(nv.STATE_NAMES)
    tbl = pd.DataFrame({
        "n": counts,
        "share": (counts / len(cases)).round(4),
    })
    say("  Actual next observed label over all prediction cases:")
    say(block(tbl))
    n_eligible = int(cases["eligible"].sum())
    say("")
    say(f"  eligible for two-class scoring (TD or PIGD)  {n_eligible:,}")
    say(f"  excluded (Indeterminate next visit)          {len(cases) - n_eligible:,}")
    say("")
    say("  Current label at the prediction point (Indeterminate is input evidence):")
    say(block(cases["current_label"].value_counts().reindex(nv.STATE_NAMES).to_frame("n")))
    say("")
    xt = pd.crosstab(cases["current_label"], cases["actual_next_label"]).reindex(
        index=nv.STATE_NAMES, columns=nv.STATE_NAMES, fill_value=0)
    say("  Current label (rows) by actual next label (columns):")
    say(block(xt))
    xt.to_csv(out / "current_by_next_label.csv")
    tbl.to_csv(out / "outcome_counts.csv")

    # ------------------------------------------------------------------ LOPO
    section("4. LEAVE-ONE-PATIENT-OUT")
    patients = data.patients[: args.limit] if args.limit else data.patients
    say(f"  folds                {len(patients):,}")
    say(f"  workers              {args.workers}")
    say("  Each fold refits every candidate on the other patients only. No")
    say("  visit of the held-out patient enters any fit, nor the direct model's")
    say("  standardisation. Visits are split by PATIENT, never individually.")

    predictions, failures = run_folds(
        data, cases, patients, cfg.seed, cfg.model["n_restarts"],
        cfg.model["emission_floor"], args.workers)

    if not len(predictions):
        raise SystemExit("every fold failed; nothing to evaluate")

    predictions = predictions.sort_values(
        ["model", "PATNO", "visit_index"]).reset_index(drop=True)
    cols = ["PATNO", "current_event", "current_years", "target_event",
            "target_years", "interval_years", "n_history", "current_label",
            "model", "P_TD", "P_PIGD", "predicted_state", "actual_next_label",
            "eligible", "two_class_defined", "is_tie", "p_TD_raw",
            "p_Indeterminate_raw", "p_PIGD_raw", "visit_index",
            "fit_train_loglik", "fit_scipy_success", "fit_proj_grad",
            "fit_converged", "fit_termination", "n_train_patients",
            "held_out_patient"]
    predictions[cols].to_csv(out / "next_visit_predictions.csv", index=False)
    failures.to_csv(out / "failed_folds.csv", index=False)

    say("")
    say(f"  folds attempted      {len(patients):,}")
    say(f"  folds completed      {predictions['held_out_patient'].nunique():,}")
    say(f"  folds failed         {len(failures):,}")
    if len(failures):
        say(block(failures, index=False))
    say(f"  prediction rows      {len(predictions):,}  "
        f"({predictions['model'].nunique()} models x {len(cases):,} cases)")
    say(f"  elapsed              {time.time() - started:.0f}s")

    # ---------------------------------------------------------------- checks
    section("5. CORRECTNESS CHECKS")
    checks = nv.correctness_checks(predictions, cases, data, failures,
                                   patients=patients)
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # --------------------------------------------------------- convergence
    section("6. CONVERGENCE AND FAILED FITS")
    conv = nv.convergence_report(predictions)
    conv.round(4).to_csv(out / "convergence.csv", index=False)
    say(block(conv.round(4), index=False))
    say("")
    say("  scipy's success flag is not the convergence criterion on its own.")
    say("  With finite-difference gradients L-BFGS-B often stops with")
    say("  ABNORMAL_TERMINATION_IN_LNSRCH at a point whose projected gradient is")
    say("  tiny; src/ppmi_subtype/hmm_stability.py documents why. A fold counts")
    say(f"  as converged when scipy succeeds OR the projected gradient is below")
    say(f"  {nv.PROJ_GRAD_TOL}. Both columns are reported so the distinction stays visible.")
    msgs = (predictions[predictions["model"] == nv.MODEL_CTHMM]
            .drop_duplicates("held_out_patient")["fit_termination"]
            .value_counts())
    say("")
    say("  CT-HMM termination messages across folds:")
    say(block(msgs.to_frame("n_folds")))

    # -------------------------------------------------------------- metrics
    section("7. MODEL COMPARISON ON IDENTICAL CASES")
    rows = [nv.two_class_metrics(g.reset_index(drop=True), seed=cfg.seed)
            for _, g in predictions.groupby("model", sort=False)]
    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    comparison.round(4).to_csv(out / "model_comparison.csv", index=False)

    show = ["model", "n_scored", "accuracy", "accuracy_ci_patient_bootstrap_low",
            "accuracy_ci_patient_bootstrap_high", "brier_two_class", "log_loss",
            "n_ties", "rule_based_probabilities"]
    say(block(comparison[show].round(4), index=False))
    say("")
    say(f"  Every model is scored on the same cases: the same "
        f"{int(comparison['n_patients'].max()):,} patients, the same "
        f"{int(comparison['n_cases_total'].max()):,} prediction opportunities,")
    say(f"  the same {int(comparison['n_eligible'].max()):,} eligible ones.")
    say("  Persistence's probabilities are a rule, not an estimate, so its log")
    say("  loss is only finite because it is clipped; its accuracy and Brier")
    say("  score are comparable, its log loss is not.")

    # Eligible-set identity: the point of the comparison.
    scored_sets = {m: set(map(tuple, nv.scored(g)[["PATNO", "visit_index"]].to_numpy()))
                   for m, g in predictions.groupby("model", sort=False)}
    sizes = {m: len(v) for m, v in scored_sets.items()}
    identical = len(set(map(frozenset, scored_sets.values()))) == 1
    say("")
    say(f"  scored-set sizes     {sizes}")
    say(f"  all models scored the identical set of cases: {identical}")
    if not identical:
        reference = max(scored_sets.values(), key=len)
        for m, v in scored_sets.items():
            if v != reference:
                say(f"    {m}: {len(reference - v):,} case(s) it could not score")

    # ------------------------------------------------------------- the best
    best_name = comparison.iloc[0]["model"]
    best = predictions[predictions["model"] == best_name].reset_index(drop=True)

    section(f"8. BEST MODEL IN DETAIL: {best_name}")
    metrics = nv.two_class_metrics(best, seed=cfg.seed)
    (out / "best_model_metrics.json").write_text(
        json.dumps({"model": best_name, **metrics}, indent=2, default=float),
        encoding="utf-8")
    say(f"  patients                       {metrics['n_patients']:,}")
    say(f"  eligible predictions scored    {metrics['n_scored']:,}")
    say(f"  ACCURACY                       {metrics['accuracy']:.4f}  "
        f"({metrics['accuracy']:.1%})")
    say(f"    95% CI, patient bootstrap    "
        f"[{metrics['accuracy_ci_patient_bootstrap_low']:.4f}, "
        f"{metrics['accuracy_ci_patient_bootstrap_high']:.4f}]")
    say(f"  Brier (two-class)              {metrics['brier_two_class']:.4f}")
    say(f"  log loss                       {metrics['log_loss']:.4f}")
    say(f"  mean P_PIGD                    {metrics['mean_P_PIGD']:.4f}")
    say(f"  observed PIGD rate             {metrics['observed_rate_PIGD']:.4f}")
    say(f"  mean p_Indeterminate_raw       {metrics['mean_p_Indeterminate_raw']:.4f}")
    say(f"  exact ties                     {metrics['n_ties']:,}")

    slugs = {_slug(m) for m in predictions["model"].unique()}
    if len(slugs) != predictions["model"].nunique():
        raise SystemExit(f"model filename slugs collide: {sorted(slugs)}")
    for name, g in predictions.groupby("model", sort=False):
        slug = _slug(name)
        nv.confusion(g).to_csv(out / f"confusion_{slug}.csv")
        nv.per_class_metrics(g).round(4).to_csv(out / f"per_class_{slug}.csv",
                                                index=False)
        nv.calibration(g).round(4).to_csv(out / f"calibration_{slug}.csv", index=False)

    cm = nv.confusion(best)
    say("")
    say("  Confusion matrix (rows predicted, columns actual):")
    say(block(cm))
    say("")
    say("  Row-normalised:")
    say(block(cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).round(3)))
    say("")
    say("  Per class:")
    say(block(nv.per_class_metrics(best).round(4), index=False))
    say("")
    say("  Calibration of P_PIGD:")
    cal = nv.calibration(best)
    say(block(cal.round(4), index=False))
    if len(cal):
        gap = float((cal["mean_P_PIGD"] - cal["observed_rate_PIGD"]).abs().mean())
        say(f"\n  Mean absolute calibration gap: {gap:.4f}")

    # ------------------------------------------------------- indeterminate
    section("9. INDETERMINATE NEXT VISITS, REPORTED SEPARATELY")
    ind_rows = []
    for name, g in predictions.groupby("model", sort=False):
        ind_rows.append({"model": name, **nv.indeterminate_report(g)})
    ind = pd.DataFrame(ind_rows)
    ind.round(4).to_csv(out / "indeterminate_report.csv", index=False)
    say(block(ind.round(4), index=False))
    say("")
    say("  These cases are not scored. The conditioning event 'next label is TD")
    say("  or PIGD' did not occur, so neither class is the right answer, and any")
    say("  rule for calling them would be a decision about the label rather than")
    say("  a measurement of the model. What each model would have said is shown")
    say("  above so the omission can be judged.")

    # ---------------------------------------------- what the question asks
    section("10. DOES LATENT STATE OR ELAPSED TIME HELP?")
    acc = comparison.set_index("model")
    pairs = [
        ("latent state, holding the observed process interval-aware",
         nv.MODEL_CTHMM, nv.MODEL_CTMC),
        ("elapsed time, holding the observed-label chain fixed",
         nv.MODEL_CTMC, nv.MODEL_DTMC),
        ("elapsed time, in the discriminative family",
         nv.MODEL_DIRECT, nv.MODEL_DIRECT_NOINT),
        ("latent generative against the discriminative model",
         nv.MODEL_CTHMM, nv.MODEL_DIRECT),
        ("any fitted model over the persistence rule",
         best_name, nv.MODEL_PERSIST),
        ("any fitted model over the null",
         best_name, nv.MODEL_MARGINAL),
    ]
    contrast_rows = []
    for what, a, b in pairs:
        if a not in acc.index or b not in acc.index or a == b:
            continue
        paired = nv.paired_accuracy_difference(
            predictions[predictions["model"] == a],
            predictions[predictions["model"] == b], seed=cfg.seed)
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
    say("  accuracy_delta is model_a minus model_b on the identical scored cases,")
    say("  with a 95% interval from resampling PATIENTS. Comparing the two models'")
    say("  separate intervals in section 7 would be the wrong test: the same cases")
    say("  score both models, so the comparison has to be paired.")
    say("  A negative brier_delta favours model_a. The two discordance counts are")
    say("  the McNemar cells. The exact McNemar p is given for reference only and")
    say("  is anti-conservative here, because it treats a patient's several")
    say("  predictions as independent when they are not.")

    # ------------------------------------------------- where the gains live
    section("11. WHERE THE DIFFERENCES LIVE")
    breakdowns = nv.accuracy_breakdowns(predictions)
    for key, frame in breakdowns.items():
        frame.round(4).to_csv(out / f"accuracy_{key}.csv", index=False)

    say(f"  By inter-visit interval (split at {nv.INTERVAL_SPLIT_YEARS} years):")
    say(block(breakdowns["by_interval"].round(4), index=False))
    say("")
    say("  PPMI's schedule is annual early and biennial later, so most gaps sit")
    say("  within a couple of months of one year. A model that reads elapsed time")
    say("  has almost nothing to read on those cases, which is why the two groups")
    say("  are reported apart rather than averaged.")
    say("")
    say("  By the label recorded at the prediction point:")
    say(block(breakdowns["by_current_label"].round(4), index=False))
    say("")
    say("  By how many of the patient's own visits the prediction could see:")
    say(block(breakdowns["by_history_length"].round(4), index=False))

    # -------------------------------------------------------- reproducibility
    section("12. PROVENANCE")
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "node": platform.node(),
        "workers": args.workers,
        "seed": cfg.seed,
        "n_restarts": cfg.model["n_restarts"],
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
        [Path(__file__), ROOT / "src/ppmi_subtype/next_visit.py",
         ROOT / "src/ppmi_subtype/hmm.py", ROOT / "src/ppmi_subtype/ctmc.py",
         ROOT / "src/ppmi_subtype/loo.py",
         ROOT / "src/ppmi_subtype/hmm_stability.py"]
    }
    for k, v in env.items():
        if k != "code_sha256":
            say(f"  {k:<20} {v}")

    summary = {
        "environment": env,
        "design": {
            "task": "next observed motor subtype, TD vs PIGD, conditional on the "
                    "next label not being Indeterminate",
            "validation": "leave-one-patient-out; whole patient held out from "
                          "every fit and from the direct model's standardisation",
            "filtering": "forward recursion only; no backward/smoothing pass and "
                         "no quantity depending on the target visit",
            "elapsed_time": "dt to the next visit is an input to the "
                            "interval-aware models",
            "tie_rule": nv.TIE_RULE,
            "indeterminate": "kept as input evidence and as a reported outcome; "
                             "never recoded and never scored",
        },
        "counts": {
            "n_patients": len(data.sequences),
            "n_cases": len(cases),
            "n_eligible": n_eligible,
            "n_indeterminate_next": len(cases) - n_eligible,
            "outcome_counts": counts.to_dict(),
            "n_folds_attempted": len(patients),
            "n_folds_failed": len(failures),
        },
        "checks": checks,
        "best_model": best_name,
        "model_comparison": comparison.to_dict(orient="records"),
        "convergence": conv.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "accuracy_breakdowns": {k: v.to_dict(orient="records")
                                for k, v in breakdowns.items()},
        "indeterminate": ind.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    # The manifest cannot hash itself, nor the shell capture of this process's
    # own stdout: driver.log is still being appended to while the manifest is
    # written, so including it guarantees a verification failure. The same
    # content is in report.txt, which is hashed.
    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
