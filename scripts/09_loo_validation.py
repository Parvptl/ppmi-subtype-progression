"""Patient-level leave-one-out validation of the progression models.

Each of the 934 patients with two or more labelled visits is held out in turn.
The models are refitted from scratch on the other 933 and then used to predict
the held-out patient's recorded subtype one visit at a time, using only that
patient's earlier visits as history.

What is being predicted is the patient's NEXT RECORDED SUBTYPE LABEL, produced
by this project's own classification rule. It is not an independently verified
diagnosis, so accuracy here measures agreement with a noisy record rather than
accuracy about disease progression.

    python scripts/09_loo_validation.py [--limit N] [--no-baselines]

Writes to outputs/loo/. The fold loop checkpoints, so an interrupted run resumes
rather than restarting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import loo, loo_plots  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def pick_examples(frame: pd.DataFrame, summary: pd.DataFrame) -> tuple[list, dict]:
    """Choose example patients by a fixed, stated rule rather than by eye.

    One patient per category below, each time taking the lowest PATNO among the
    candidates so the choice is deterministic and not selected for looking good.
    """
    eligible = set(summary.loc[summary["n_predictions"] >= 4, "PATNO"])
    by_patient = {p: g.sort_values("visit_index")
                  for p, g in frame.groupby("PATNO") if p in eligible}

    def first(pred) -> object | None:
        hits = sorted(p for p, g in by_patient.items() if pred(g))
        return hits[0] if hits else None

    def changed(g) -> bool:
        seq = [g["current"].iloc[0]] + g["actual"].tolist()
        return len(set(seq)) > 1

    def reverses(g) -> bool:
        seq = [g["current"].iloc[0]] + g["actual"].tolist()
        collapsed = [s for k, s in enumerate(seq) if k == 0 or s != seq[k - 1]]
        return any(collapsed[k] == collapsed[k + 2] for k in range(len(collapsed) - 2))

    rules = [
        ("stable throughout, model correct every time",
         lambda g: not changed(g) and g["correct"].all()),
        ("recorded change from TD to PIGD",
         lambda g: any((g["current"] == "TD") & (g["actual"] == "PIGD"))),
        ("subtype reverses back to an earlier state", reverses),
        ("passes through Indeterminate",
         lambda g: "Indeterminate" in set(g["actual"]) | set(g["current"])),
    ]

    chosen, notes = [], {}
    for note, rule in rules:
        p = first(lambda g, r=rule: r(g) and g["PATNO"].iloc[0] not in chosen)
        if p is not None:
            chosen.append(p)
            notes[p] = note

    # The patient the model does worst on, and the one with the most changes.
    worst = summary[summary["PATNO"].isin(eligible) & ~summary["PATNO"].isin(chosen)]
    if len(worst):
        p = worst.sort_values(["accuracy", "PATNO"]).iloc[0]["PATNO"]
        chosen.append(p)
        notes[p] = "lowest accuracy of any patient with four or more predictions"

    changes = {p: sum(g["current"].to_numpy() != g["actual"].to_numpy())
               for p, g in by_patient.items() if p not in chosen}
    if changes:
        p = max(sorted(changes), key=lambda k: changes[k])
        chosen.append(p)
        notes[p] = f"most recorded changes ({changes[p]}) of any held-out patient"

    return chosen, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N folds (for a smoke test)")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = cfg.path("outputs") / "loo"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {v: k for k, v in cfg.subtype.labels.items()}
    states = [cfg.subtype.labels[s] for s in loo.STATE_NAMES]

    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    data = loo.prepare(panel, states)

    # ---------------------------------------------------------------- design
    section("1. DESIGN")
    say("  Every patient is held out once. The models are refitted on the other")
    say("  933 patients and then predict the held-out patient's visits in order:")
    say("  visit 2 is predicted from visit 1, visit 3 from visits 1-2, and so on.")
    say("  The visit being predicted is never available as input, and no visit of")
    say("  the held-out patient enters the fit.")
    say("")
    say("  The target is the NEXT RECORDED SUBTYPE LABEL from our own")
    say("  classification rule, not an independent clinical diagnosis.")
    say("")
    say(f"  patients with two or more labelled visits  {len(data.sequences):,}")
    say(f"  prediction opportunities (transitions)     {data.n_transitions:,}")
    say(f"  emission floor                             {cfg.model['emission_floor']}")
    say(f"  restarts per fit                           {cfg.model['n_restarts']}")
    say(f"  seed                                       {cfg.seed}")

    if len(data.sequences) != 934 or data.n_transitions != 3361:
        raise SystemExit(
            f"expected 934 patients and 3,361 transitions; "
            f"got {len(data.sequences)} and {data.n_transitions}")

    # ------------------------------------------------------------------ run
    section("2. RUNNING THE FOLDS")
    checkpoint = out_dir / "loo_predictions_partial.csv"
    raw = loo.run_loo(
        data, seed=cfg.seed, n_restarts=cfg.model["n_restarts"],
        emission_floor=cfg.model["emission_floor"],
        checkpoint=checkpoint, checkpoint_every=25,
        limit=args.limit, with_baselines=not args.no_baselines,
    )
    preds = loo.annotate(raw, data, names)
    preds.to_csv(out_dir / "loo_predictions.csv", index=False)

    hmm_preds = preds[preds["model"] == "HMM"].reset_index(drop=True)

    # -------------------------------------------------------------- checks
    section("3. LEAKAGE AND CONSISTENCY CHECKS")
    expected_patients = len(data.sequences) if not args.limit else args.limit
    expected_preds = (data.n_transitions if not args.limit
                      else int(sum(len(s["gaps"]) for s in data.sequences[:args.limit])))

    cm = loo.confusion(hmm_preds)
    argmax_names = np.array(loo.STATE_NAMES)[
        hmm_preds[loo.PROB_COLS].to_numpy().argmax(axis=1)]

    checks = {
        "every patient held out exactly once":
            hmm_preds["held_out_patient"].nunique() == expected_patients,
        "held-out patient is the patient being predicted":
            bool((hmm_preds["held_out_patient"] == hmm_preds["PATNO"]).all()),
        "every fold trained on all other patients":
            set(hmm_preds["n_train_patients"].unique()) == {len(data.sequences) - 1},
        "prediction count equals transition count":
            len(hmm_preds) == expected_preds,
        "history is strictly the visits before the target":
            bool((hmm_preds["n_history"] == hmm_preds["visit_index"] + 1).all()),
        "target visit is always later than the current visit":
            bool((hmm_preds["target_years"] >= hmm_preds["current_years"]).all()),
        "probabilities sum to one":
            bool(np.allclose(hmm_preds[loo.PROB_COLS].sum(axis=1), 1.0, atol=1e-8)),
        "predicted class is the argmax":
            bool((hmm_preds["predicted"].to_numpy() == argmax_names).all()),
        "confusion matrix totals equal prediction count":
            int(cm.to_numpy().sum()) == len(hmm_preds),
        "every fit converged":
            bool(hmm_preds["train_converged"].all()),
    }

    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # ------------------------------------------------------------- metrics
    section("4. HEADLINE RESULT (HMM)")
    metrics = loo.overall_metrics(hmm_preds, seed=cfg.seed)
    loo.save_json(metrics, out_dir / "loo_overall_metrics.json")

    say(f"  patients                       {metrics['n_patients']:,}")
    say(f"  predictions                    {metrics['n_predictions']:,}")
    say(f"  correct                        {metrics['n_correct']:,}")
    say(f"  ACCURACY                       {metrics['accuracy']:.4f}  "
        f"({metrics['accuracy']:.1%})")
    say(f"    95% CI, patient bootstrap    "
        f"[{metrics['accuracy_ci_patient_bootstrap_low']:.4f}, "
        f"{metrics['accuracy_ci_patient_bootstrap_high']:.4f}]")
    say(f"    95% CI, Wilson (no clustering) "
        f"[{metrics['accuracy_ci_wilson_low']:.4f}, {metrics['accuracy_ci_wilson_high']:.4f}]")
    say(f"  mean per-patient accuracy      {metrics['mean_per_patient_accuracy']:.4f}")
    say(f"  median per-patient accuracy    {metrics['median_per_patient_accuracy']:.4f}")
    say(f"  log-likelihood per prediction  {metrics['loglik_per_prediction']:.4f}")
    say(f"  Brier score                    {metrics['brier']:.4f}")
    say(f"  mean confidence                {metrics['mean_confidence']:.4f}")
    say("\n  The patient bootstrap interval is the one to quote: predictions are")
    say("  clustered within patients, so treating them as independent understates")
    say("  the uncertainty.")

    section("5. CONFUSION MATRIX (rows predicted, columns actual)")
    cm.to_csv(out_dir / "loo_confusion_matrix.csv")
    say("    " + cm.to_string().replace("\n", "\n    "))
    row_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    row_norm.round(4).to_csv(out_dir / "loo_confusion_matrix_rownorm.csv")
    say("\n  Row-normalised:")
    say("    " + row_norm.round(3).to_string().replace("\n", "\n    "))

    per_class = loo.per_class_metrics(hmm_preds)
    per_class.round(4).to_csv(out_dir / "loo_per_class_metrics.csv", index=False)
    say("\n  Per class:")
    say("    " + per_class.round(3).to_string(index=False).replace("\n", "\n    "))

    section("6. ACCURACY BY TRANSITION TYPE")
    trans = loo.transition_accuracy(hmm_preds)
    trans.round(4).to_csv(out_dir / "loo_transition_metrics.csv", index=False)
    say("    " + trans.round(3).to_string(index=False).replace("\n", "\n    "))

    section("7. CALIBRATION")
    cal = loo.calibration(hmm_preds)
    cal.round(4).to_csv(out_dir / "loo_calibration.csv", index=False)
    say("    " + cal.round(3).to_string(index=False).replace("\n", "\n    "))
    gap = float((cal["mean_confidence"] - cal["observed_accuracy"]).abs().mean())
    say(f"\n  Mean absolute gap between confidence and accuracy: {gap:.4f}")

    section("8. BY CURRENT STATE AND BY HISTORY LENGTH")
    by_state = hmm_preds.groupby("current")["correct"].agg(["mean", "size"])
    say("  Accuracy by the subtype the patient is currently recorded in:")
    say("    " + by_state.round(4).to_string().replace("\n", "\n    "))
    by_state.round(4).to_csv(out_dir / "loo_accuracy_by_current_state.csv")

    hist = hmm_preds.assign(bucket=hmm_preds["n_history"].clip(upper=5)) \
        .groupby("bucket")["correct"].agg(["mean", "size"])
    say("\n  Accuracy by number of the patient's own prior visits (5 = 5 or more):")
    say("    " + hist.round(4).to_string().replace("\n", "\n    "))
    hist.round(4).to_csv(out_dir / "loo_accuracy_by_history.csv")

    section("9. PER-PATIENT VARIATION")
    summary = loo.patient_summary(hmm_preds)
    summary.round(4).to_csv(out_dir / "loo_patient_summary.csv", index=False)
    acc = summary["accuracy"]
    say(f"  patients                     {len(summary):,}")
    say(f"  mean per-patient accuracy    {acc.mean():.4f}")
    say(f"  median                       {acc.median():.4f}")
    say(f"  sd                           {acc.std():.4f}")
    say(f"  IQR                          {acc.quantile(.25):.4f} to {acc.quantile(.75):.4f}")
    say(f"  patients at 0%               {(acc == 0).sum():,} "
        f"({(acc == 0).mean():.1%})")
    say(f"  patients at 100%             {(acc == 1).sum():,} "
        f"({(acc == 1).mean():.1%})")
    single = summary["n_predictions"] == 1
    say(f"  patients with a single prediction, so only 0% or 100% is possible: "
        f"{single.sum():,} ({single.mean():.1%})")

    section("10. MODEL COMPARISON ON THE SAME PREDICTIONS")
    rows = []
    for model, g in preds.groupby("model"):
        m = loo.overall_metrics(g.reset_index(drop=True), seed=cfg.seed)
        m["model"] = model
        rows.append(m)
    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    cols = ["model", "n_predictions", "accuracy",
            "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high",
            "mean_per_patient_accuracy", "loglik_per_prediction", "brier"]
    comparison = comparison[cols]
    comparison.round(4).to_csv(out_dir / "loo_model_comparison.csv", index=False)
    say("    " + comparison.round(4).to_string(index=False).replace("\n", "\n    "))
    say("\n  Persistence carries no fitted parameters: it always names the current")
    say("  subtype. Its probabilities are a rule, not an estimate, so its Brier")
    say("  score and log-likelihood are not comparable with the fitted models.")

    # -------------------------------------------------------------- figures
    section("11. FIGURES")
    chosen, notes = pick_examples(hmm_preds, summary)
    pd.DataFrame({"PATNO": chosen, "reason": [notes[p] for p in chosen]}).to_csv(
        out_dir / "loo_example_patients.csv", index=False)

    written = [
        loo_plots.confusion_matrix(cm, fig_dir / "loo_confusion_matrix.png"),
        loo_plots.confusion_matrix(cm, fig_dir / "loo_confusion_matrix_normalized.png",
                                   normalize=True),
        loo_plots.accuracy_by_current_state(
            hmm_preds, fig_dir / "loo_accuracy_by_current_state.png"),
        loo_plots.accuracy_by_history(
            hmm_preds, fig_dir / "loo_accuracy_by_history_length.png"),
        loo_plots.calibration_plot(cal, fig_dir / "loo_calibration.png"),
        loo_plots.confidence_correct_vs_incorrect(
            hmm_preds, fig_dir / "loo_confidence_correct_vs_incorrect.png"),
        loo_plots.probability_by_actual(
            hmm_preds, fig_dir / "loo_probability_by_actual_class.png"),
        loo_plots.transition_accuracy_plot(
            trans, fig_dir / "loo_transition_accuracy.png"),
        loo_plots.observed_vs_predicted(
            hmm_preds, fig_dir / "loo_observed_vs_predicted.png"),
        loo_plots.patient_accuracy_distribution(
            summary, fig_dir / "loo_patient_accuracy_distribution.png"),
        loo_plots.accuracy_vs_visits(
            summary, fig_dir / "loo_accuracy_vs_num_visits.png"),
        loo_plots.accuracy_over_followup(
            hmm_preds, fig_dir / "loo_accuracy_over_followup.png"),
        loo_plots.example_trajectories(
            hmm_preds, chosen, fig_dir / "loo_example_trajectories.png", notes),
        loo_plots.model_comparison(
            pd.DataFrame(rows), fig_dir / "loo_model_comparison.png"),
    ]
    for p in written:
        say(f"  {p.relative_to(cfg.root)}")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
