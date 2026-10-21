"""Diagnostic: is the latent-state deficit the features, or the state bottleneck?

Script 62 found the latent clinical-state model markedly worse than the
subtype-history CT-HMM at predicting the next observed TD/PIGD label. Two very
different explanations fit that result:

  (a) the ten clinical measurements do not carry next-label information, or
  (b) they do, and compressing them into a handful of discrete latent states
      destroys it.

The difference matters: (a) says rich clinical data is not useful for this
task, (b) says the *architecture* is not, which is a statement about the model
and not about the data. They are distinguished by fitting a model that uses the
same ten measurements with **no latent bottleneck at all** --- a direct
two-class logistic regression on the measurements themselves --- and scoring it
on the same held-out cases.

This reuses :func:`next_visit.fit_direct` unchanged, which is what makes the
comparison fair: the penalty, the training-only standardisation and the
conditional two-class target are identical to the previous experiment's direct
model, only the feature columns differ.

    python scripts/63_clinical_features_direct.py --out <dir> --selection <dir>
                                                  --lopo <script 62 run dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from ppmi_subtype import clinical_states as cs  # noqa: E402
from ppmi_subtype import next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

split_patients = import_module("61_clinical_state_selection").split_patients

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

MODEL_DIRECT = "Direct logistic on the 10 clinical measurements"
MODEL_DIRECT_RATIO = "Direct logistic on tremor and PIGD scores only"

#: The clinical measurements as prediction features, plus the elapsed time,
#: which the interval-aware models also receive.
CLINICAL_FEATURES = cs.FEATURE_NAMES + ("log_interval",)

#: The two measurements the subtype label is actually defined from. If these
#: alone recover the baseline's performance, the label information was never
#: distributed across the ten features in the first place.
RATIO_FEATURES = ("tremor", "pigd", "log_interval")

lines: list[str] = []


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


def build_cases(sequences) -> pd.DataFrame:
    """One row per prediction opportunity, with current-visit measurements.

    Every column is a measurement taken at the current visit or the elapsed
    time to the next one. Nothing from the next visit enters, and the target is
    the next visit's recorded label.
    """
    rows = []
    for seq in sequences:
        X, gaps, labels = seq["X"], seq["gaps"], seq["labels"]
        for k, gap in enumerate(gaps):
            row = {
                "PATNO": seq["PATNO"],
                "visit_index": int(seq["visit_index"][k]),
                "n_history": k + 1,
                "current_event": seq["events"][k],
                "current_years": float(seq["times"][k]),
                "target_event": seq["events"][k + 1],
                "target_years": float(seq["times"][k + 1]),
                "interval_years": float(gap),
                "log_interval": float(np.log(gap)),
                "current_label": cs.STATE_NAMES[int(labels[k]) - 1],
                "actual_next_label": cs.STATE_NAMES[int(labels[k + 1]) - 1],
                "target_index": int(labels[k + 1]) - 1,
            }
            for j, name in enumerate(cs.FEATURE_NAMES):
                row[name] = float(X[k, j])
            rows.append(row)

    cases = pd.DataFrame(rows)
    cases["eligible"] = cases["target_index"].isin([nv.TD, nv.PIGD])
    # fit_direct standardises internally but cannot read NaN. Missingness is
    # under 3% per feature here and a visit needs 8 of 10 features to exist at
    # all, so the few gaps are filled with the TRAINING median inside the fold
    # loop -- never a global statistic.
    return cases


def fill_missing(train: pd.DataFrame, frames, features) -> list[pd.DataFrame]:
    """Fill NaN with the training-fold median. Returns copies, train first."""
    medians = train[list(features)].median()
    return [f.assign(**{c: f[c].fillna(medians[c]) for c in features})
            for f in frames]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--lopo", required=True,
                    help="script 62's run dir, for the latent model's predictions")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    section("1. INPUT")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected")
    say(f"  panel md5         {panel_md5}  [MATCH]")

    panel = pd.read_csv(panel_path, low_memory=False)
    clinical = cs.build_clinical_panel(cfg, panel)
    sequences = cs.build_sequences(clinical)
    cases = build_cases(sequences)

    dev, evaluation = split_patients([s["PATNO"] for s in sequences], cfg.seed)
    sel_dev = set(pd.read_csv(
        Path(args.selection) / "development_patients.csv")["PATNO"])
    if sel_dev != set(dev):
        raise SystemExit("reproduced development split differs from script 61's")
    say(f"  evaluation patients {len(evaluation):,}  (same split as scripts 61/62)")
    say(f"  prediction cases    {len(cases):,}")
    say("")
    say("  The question: the latent-state model was much worse than the")
    say("  subtype-history baseline. Was that the measurements, or the")
    say("  compression of them into a few discrete states? A model on the same")
    say("  measurements with NO latent layer separates the two.")

    # ------------------------------------------------------------------ LOPO
    section("2. LOPO WITH NO LATENT LAYER")
    say("  Per fold: hold the patient out, fit the median fill and the")
    say("  L2-penalised logistic regression on the other patients' eligible")
    say("  cases only, predict the held-out patient's visits.")
    say(f"  features: {list(CLINICAL_FEATURES)}")

    specs = {MODEL_DIRECT: CLINICAL_FEATURES, MODEL_DIRECT_RATIO: RATIO_FEATURES}
    collected: dict[str, list[pd.DataFrame]] = {k: [] for k in specs}
    failures = []

    for n, patno in enumerate(evaluation, start=1):
        train = cases[cases["PATNO"] != patno]
        test = cases[cases["PATNO"] == patno]
        if not len(test):
            continue
        for name, features in specs.items():
            try:
                tr, te = fill_missing(train, [train, test], features)
                fit = nv.fit_direct(tr, features=features)
                p_pigd = fit.predict_p_pigd(te)
                frame = te[["PATNO", "visit_index", "n_history", "current_event",
                            "current_years", "target_event", "target_years",
                            "interval_years", "current_label",
                            "actual_next_label", "eligible"]].copy()
                frame["model"] = name
                frame["p_TD_raw"] = 1.0 - p_pigd
                frame["p_Indeterminate_raw"] = 0.0
                frame["p_PIGD_raw"] = p_pigd
                frame["P_TD"] = 1.0 - p_pigd
                frame["P_PIGD"] = p_pigd
                frame["two_class_defined"] = True
                frame["predicted_state"] = nv.predicted_state(1.0 - p_pigd, p_pigd)
                frame["is_tie"] = (1.0 - p_pigd) == p_pigd
                frame["fit_converged"] = bool(
                    fit.converged or fit.proj_grad < nv.PROJ_GRAD_TOL)
                frame["held_out_patient"] = patno
                frame["n_train_patients"] = int(train["PATNO"].nunique())
                collected[name].append(frame)
            except Exception as exc:                # noqa: BLE001 -- recorded
                failures.append({"held_out_patient": patno, "model": name,
                                 "error": f"{type(exc).__name__}: {exc}"})
        if n % 200 == 0:
            say(f"  fold {n:,}/{len(evaluation):,}")

    direct = pd.concat([f for v in collected.values() for f in v],
                       ignore_index=True)
    fails = pd.DataFrame(failures, columns=["held_out_patient", "model", "error"])
    fails.to_csv(out / "failed_folds.csv", index=False)
    say("")
    say(f"  folds completed   {direct['held_out_patient'].nunique():,}")
    say(f"  folds failed      {len(fails):,}")
    say(f"  non-converged     "
        f"{int((~direct.drop_duplicates(['model','held_out_patient'])['fit_converged']).sum()):,}")
    say(f"  elapsed           {time.time() - started:.0f}s")

    # ------------------------------------------------------- bring in the rest
    section("3. ON IDENTICAL CASES, AGAINST THE LATENT MODEL AND THE BASELINE")
    prior = pd.read_csv(Path(args.lopo) / "clinical_state_predictions.csv",
                        low_memory=False)
    keep = [c for c in direct.columns if c in prior.columns]
    all_preds = pd.concat([prior[keep], direct[keep]], ignore_index=True)

    key = ["PATNO", "current_event", "target_event"]
    common = None
    for _, g in all_preds.groupby("model", sort=False):
        s = set(map(tuple, g[key].to_numpy()))
        common = s if common is None else (common & s)
    mask = np.array([tuple(r) in common for r in all_preds[key].to_numpy()])
    dropped = all_preds[~mask]
    all_preds = all_preds[mask].reset_index(drop=True)
    if len(dropped):
        say(f"  cases dropped to make the sets identical: {len(dropped):,}")
        dropped.to_csv(out / "cases_dropped_for_pairing.csv", index=False)

    for name, g in all_preds.groupby("model", sort=False):
        say(f"  {name:<46} {len(g):,} cases")

    all_preds.to_csv(out / "direct_clinical_predictions.csv", index=False)

    rows = [nv.two_class_metrics(g.reset_index(drop=True), seed=cfg.seed)
            for _, g in all_preds.groupby("model", sort=False)]
    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    comparison.round(4).to_csv(out / "model_comparison.csv", index=False)
    show = ["model", "n_scored", "accuracy", "accuracy_ci_patient_bootstrap_low",
            "accuracy_ci_patient_bootstrap_high", "brier_two_class", "log_loss"]
    say("")
    say(block(comparison[show].round(4), index=False))

    # ------------------------------------------------------------- contrasts
    section("4. PAIRED CONTRASTS: WHAT THE BOTTLENECK COSTS")
    present = set(all_preds["model"])
    latent = next((m for m in present if m.startswith("Clinical-state")), None)
    baseline = next((m for m in present if m.startswith("Subtype-history")), None)
    acc = comparison.set_index("model")

    pairs = [("same measurements, no latent bottleneck", MODEL_DIRECT, latent),
             ("the ten measurements against the subtype-history baseline",
              MODEL_DIRECT, baseline),
             ("only the two label-defining scores, against all ten",
              MODEL_DIRECT_RATIO, MODEL_DIRECT),
             ("only the two label-defining scores, against the baseline",
              MODEL_DIRECT_RATIO, baseline)]
    contrast_rows = []
    for what, a, b in pairs:
        if a not in present or b is None or b not in present:
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
              "brier_delta", "mcnemar_exact_p_ignores_clustering"]
    say(block(contrasts[show_c].round(4), index=False))

    section("5. BY CURRENT LABEL -- WHERE THE LATENT MODEL LOST ITS GROUND")
    breakdowns = nv.accuracy_breakdowns(all_preds)
    for k, f in breakdowns.items():
        f.round(4).to_csv(out / f"accuracy_{k}.csv", index=False)
    say(block(breakdowns["by_current_label"].round(4), index=False))
    say("")
    rate = (all_preds[all_preds["eligible"] & all_preds["two_class_defined"]]
            .groupby("model")
            .apply(lambda g: pd.Series({
                "says_PIGD": float((g["predicted_state"] == "PIGD").mean()),
                "actual_PIGD": float((g["actual_next_label"] == "PIGD").mean()),
                "mean_P_PIGD": float(g["P_PIGD"].mean()),
            }), include_groups=False)
            .reset_index())
    rate.round(4).to_csv(out / "pigd_call_rate.csv", index=False)
    say("  How often each model is willing to say PIGD at all:")
    say(block(rate.round(4), index=False))

    section("6. CHECKS")
    defined = all_preds[all_preds["two_class_defined"]]
    p_td = defined["P_TD"].to_numpy(float)
    p_pigd = defined["P_PIGD"].to_numpy(float)
    d = all_preds[all_preds["model"] == MODEL_DIRECT]
    checks = {
        "probabilities finite and within [0, 1]":
            bool(np.isfinite(p_td).all() and (p_td >= 0).all() and (p_td <= 1).all()),
        "P_TD + P_PIGD == 1 within tolerance":
            bool(np.allclose(p_td + p_pigd, 1.0, atol=1e-12)),
        "predicted class follows the documented rule":
            bool((defined["predicted_state"].to_numpy()
                  == nv.predicted_state(p_td, p_pigd)).all()),
        "every model scored the identical case set":
            len({frozenset(map(tuple, g[key].to_numpy()))
                 for _, g in all_preds.groupby("model", sort=False)}) == 1,
        "held-out patient is the patient being predicted":
            bool((d["held_out_patient"] == d["PATNO"]).all()),
        "no evaluation patient is in the development set":
            set(d["PATNO"]).isdisjoint(set(dev)),
        "target visit strictly later than the current visit":
            bool((all_preds["target_years"] > all_preds["current_years"]).all()),
        "Indeterminate next visits excluded from scoring":
            int(nv.scored(all_preds)["actual_next_label"]
                .isin(["Indeterminate"]).sum()) == 0,
        "no fold failed":
            len(fails) == 0,
    }
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    section("7. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "platform": platform.platform(),
           "node": platform.node(), "seed": cfg.seed, "panel_md5": panel_md5,
           "selection_run": str(args.selection), "lopo_run": str(args.lopo),
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<18} {v}")

    (out / "summary.json").write_text(json.dumps({
        "environment": env,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                        [Path(__file__), ROOT / "src/ppmi_subtype/clinical_states.py",
                         ROOT / "src/ppmi_subtype/next_visit.py"]},
        "checks": checks,
        "model_comparison": comparison.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "pigd_call_rate": rate.to_dict(orient="records"),
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
