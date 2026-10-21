"""Choose the latent state count for the clinical-state model, on development patients only.

Why this is a separate script
-----------------------------
The state count has to be chosen without consulting the outcomes of the
patients the final model is evaluated on. Choosing it inside every LOPO fold
would be the cleanest answer but costs a state-count sweep per fold, which is
hours of compute for a quantity that barely moves between folds. Instead the
patients are split once, by PATNO, into:

    development  30%  --- used here, and NOWHERE in the final evaluation
    evaluation   70%  --- untouched by this script, LOPO-evaluated in script 62

The split is deterministic given the seed, and script 62 reproduces it from
the same seed and asserts the two sets are disjoint. So no held-out outcome,
and no held-out patient's data at all, informs the state count.

The criterion is the **next-visit predictive log-density** under filtering,
averaged over an inner 4-fold patient-level cross-validation of the
development set. That is the quantity the model is actually for, and unlike
BIC it does not assume the visits within a patient are independent
observations --- which they are not, and which makes BIC over-select here.
BIC is reported alongside for comparison, not used.

    python scripts/61_clinical_state_selection.py --out <dir> [--workers N]
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
from ppmi_subtype.config import load_config  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

#: Share of patients reserved for choosing the state count. They are excluded
#: from the final evaluation entirely, so this is the price of a clean
#: selection rather than a tuning set that is later scored.
DEVELOPMENT_SHARE = 0.30

#: Candidate state counts, stopping at eight. The stopping point is a measured
#: compute limit, not a modelling opinion: one LOPO fold on 933 patients takes
#: 10s of EM at four states, 81s at six and 295s at eight, so the full 654-fold
#: evaluation costs roughly 0.3h, 2.8h and 10h at three starts per fold.
#: Counts above eight cannot be evaluated the way this experiment evaluates, so
#: sweeping them would only produce a preferred count no held-out result could
#: ever be reported for.
#: Severson et al. reported eight states; eight is the top of this range so
#: that count can be tested, not so it can be confirmed.
STATE_COUNTS = (2, 3, 4, 5, 6, 8)

N_INNER_FOLDS = 4

#: EM starts per fit. Four was not enough: the spread in final log-likelihood
#: across starts exceeded 1,000 nats at state counts above three, meaning
#: single-start fits land in very different optima and a state-count
#: comparison built on them compares optimiser luck. The spread at the chosen
#: count is reported, not assumed away.
N_STARTS = 8

#: Parsimony rule. The criterion flattens rather than peaking, so taking the
#: arithmetic maximum chases fold noise and the top of the range. Instead the
#: smallest state count whose mean is within one standard error (across inner
#: folds) of the best mean is chosen -- the standard one-standard-error rule.
#: It is fixed in advance and uses only development data.
USE_ONE_SE_RULE = True

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


def split_patients(patients, seed: int, dev_share: float = DEVELOPMENT_SHARE):
    """Deterministic development / evaluation split by patient.

    Script 62 calls this with the same arguments to recover the evaluation set,
    so the rule lives in one place and the two scripts cannot drift apart.
    """
    p = np.array(sorted(patients))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(p))
    n_dev = int(round(dev_share * len(p)))
    return sorted(p[order[:n_dev]].tolist()), sorted(p[order[n_dev:]].tolist())


def _init(sequences, fold_of) -> None:
    _WORK.update(sequences=sequences, fold_of=fold_of)


def _task(args):
    """One (state count, inner fold) cell: fit on the rest, score the fold."""
    n_states, fold, seed = args
    sequences, fold_of = _WORK["sequences"], _WORK["fold_of"]

    train = [s for s in sequences if fold_of[s["PATNO"]] != fold]
    test = [s for s in sequences if fold_of[s["PATNO"]] == fold]
    try:
        # Standardisation is refitted on this cell's training patients only.
        std = cs.Standardiser.fit(train)
        fit, records = cs.fit_em_multistart(
            std.apply(train), n_states, seed=seed, n_starts=N_STARTS)
        score = cs.predictive_score(fit, std.apply(test))
        return {
            "n_states": n_states, "fold": fold, "error": "",
            "train_loglik": fit.loglik, "train_bic": fit.bic(),
            "n_params": fit.n_params, "n_iter": fit.n_iter,
            "converged": fit.converged,
            "loglik_spread_across_starts":
                records[0]["loglik_spread_across_starts"],
            "n_train_patients": len(train), "n_test_patients": len(test),
            **score,
        }
    except Exception as exc:                        # noqa: BLE001 -- recorded
        return {"n_states": n_states, "fold": fold,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    section("1. INPUT AND FEATURE REPRESENTATION")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    say(f"  panel        {panel_path.relative_to(ROOT)}")
    say(f"  md5          {panel_md5}")
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}")
    say(f"  expected     {EXPECTED_PANEL_MD5}  [MATCH]")

    panel = pd.read_csv(panel_path, low_memory=False)
    clinical = cs.build_clinical_panel(cfg, panel)
    sequences = cs.build_sequences(clinical)

    inventory = cs.feature_inventory(clinical)
    inventory.round(4).to_csv(out / "feature_inventory.csv", index=False)
    say("")
    say(f"  clinical visits      {len(clinical):,}")
    say(f"  patients             {clinical['PATNO'].nunique():,}")
    say(f"  patients with >= 2   {len(sequences):,}")
    say(f"  transitions          {sum(len(s['gaps']) for s in sequences):,}")
    say(f"  features             {len(cs.FEATURES)}")
    say(f"  visit kept if it carries >= {cs.MIN_FEATURES_PER_VISIT} of "
        f"{len(cs.FEATURES)} features")
    say("")
    say(block(inventory[["feature", "domain", "transform", "coverage",
                         "mean", "sd", "min", "max"]].round(3), index=False))
    say("")
    say("  Missing measurements are MARGINALISED, not imputed: with a diagonal")
    say("  covariance the emission log-density is a sum over features, so an")
    say("  unrecorded feature is dropped from that sum exactly. No imputer is")
    say("  fitted anywhere, which removes that leakage surface entirely.")
    say("")
    say("  No dimensionality reduction is applied. Ten features with pairwise")
    say("  correlations peaking at 0.65 are not redundant, and a projection")
    say("  would cost the per-feature state profiles the interpretation needs.")

    corr = clinical[list(cs.FEATURE_NAMES)].corr()
    corr.round(3).to_csv(out / "feature_correlation.csv")

    # ------------------------------------------------------------- the split
    section("2. DEVELOPMENT / EVALUATION SPLIT BY PATIENT")
    all_patients = [s["PATNO"] for s in sequences]
    dev, evaluation = split_patients(all_patients, cfg.seed)
    dev_set = set(dev)

    say(f"  patients total       {len(all_patients):,}")
    say(f"  development (30%)    {len(dev):,}   <- state count chosen here")
    say(f"  evaluation  (70%)    {len(evaluation):,}   <- untouched by this script")
    say(f"  disjoint             {set(dev).isdisjoint(set(evaluation))}")
    say(f"  seed                 {cfg.seed}")
    pd.DataFrame({"PATNO": dev, "role": "development"}).to_csv(
        out / "development_patients.csv", index=False)
    pd.DataFrame({"PATNO": evaluation, "role": "evaluation"}).to_csv(
        out / "evaluation_patients.csv", index=False)

    dev_sequences = [s for s in sequences if s["PATNO"] in dev_set]
    say(f"  development transitions {sum(len(s['gaps']) for s in dev_sequences):,}")

    # -------------------------------------------------------------- sweep
    section("3. INNER CROSS-VALIDATION OVER STATE COUNTS")
    fold_of = cs.patient_folds([s["PATNO"] for s in dev_sequences],
                               N_INNER_FOLDS, cfg.seed)
    say(f"  candidate state counts  {list(STATE_COUNTS)}")
    say(f"  inner folds             {N_INNER_FOLDS} (patient-level)")
    say(f"  EM starts per fit       {N_STARTS}")
    say("  Criterion: mean next-visit predictive log-density on the inner")
    say("  held-out patients, from filtering only.")

    tasks = [(k, f, cfg.seed) for k in STATE_COUNTS for f in range(N_INNER_FOLDS)]
    if args.workers > 1:
        with mp.get_context("fork").Pool(
                args.workers, initializer=_init,
                initargs=(dev_sequences, fold_of)) as pool:
            results = pool.map(_task, tasks)
    else:
        _init(dev_sequences, fold_of)
        results = [_task(t) for t in tasks]

    cells = pd.DataFrame(results)
    cells.to_csv(out / "state_count_cells.csv", index=False)
    failed = cells[cells["error"] != ""]
    if len(failed):
        say("")
        say(f"  FAILED CELLS: {len(failed)}")
        say(block(failed[["n_states", "fold", "error"]], index=False))
    cells = cells[cells["error"] == ""]
    if not len(cells):
        raise SystemExit("every selection cell failed")

    summary = (cells.groupby("n_states")
               .agg(n_cells=("fold", "size"),
                    mean_heldout_loglik=("mean_next_obs_loglik", "mean"),
                    sd_heldout_loglik=("mean_next_obs_loglik", "std"),
                    mean_heldout_loglik_per_feature=(
                        "mean_next_obs_loglik_per_feature", "mean"),
                    mean_train_bic=("train_bic", "mean"),
                    mean_n_iter=("n_iter", "mean"),
                    all_converged=("converged", "all"),
                    max_start_spread=("loglik_spread_across_starts", "max"),
                    n_params=("n_params", "first"))
               .reset_index())
    summary.round(4).to_csv(out / "state_count_summary.csv", index=False)
    say("")
    say(block(summary.round(3), index=False))

    summary["se_heldout_loglik"] = (summary["sd_heldout_loglik"]
                                    / np.sqrt(summary["n_cells"]))
    arg_best = int(summary["mean_heldout_loglik"].idxmax())
    best_mean = float(summary.loc[arg_best, "mean_heldout_loglik"])
    best_se = float(summary.loc[arg_best, "se_heldout_loglik"])
    arg_max_states = int(summary.loc[arg_best, "n_states"])

    threshold = best_mean - best_se
    eligible = summary[summary["mean_heldout_loglik"] >= threshold]
    chosen = (int(eligible["n_states"].min()) if USE_ONE_SE_RULE and len(eligible)
              else arg_max_states)
    bic_pick = int(summary.loc[summary["mean_train_bic"].idxmin(), "n_states"])
    summary.round(4).to_csv(out / "state_count_summary.csv", index=False)

    say("")
    say(f"  best mean held-out log-density  {best_mean:.4f} at "
        f"{arg_max_states} states (se {best_se:.4f})")
    say(f"  one-standard-error threshold    {threshold:.4f}")
    say(f"  counts within one SE of best    "
        f"{sorted(eligible['n_states'].tolist())}")
    say(f"  CHOSEN STATE COUNT (smallest within one SE):          {chosen}")
    say(f"  Arithmetic maximum would have chosen:                 {arg_max_states}")
    say(f"  BIC would have chosen:                                {bic_pick}")
    if chosen != arg_max_states:
        say("  The criterion flattens rather than peaking, so the arithmetic")
        say("  maximum tracks fold noise and the top of the tested range. The")
        say("  parsimony rule takes the smallest count that is statistically")
        say("  indistinguishable from the best.")
    if arg_max_states == max(STATE_COUNTS):
        say("  WARNING: the best mean sits at the TOP of the tested range, so the")
        say("  criterion may still be improving beyond it. The parsimony rule is")
        say("  what keeps the choice away from that boundary.")
    if bic_pick != chosen:
        say("  They disagree. BIC treats a patient's repeated visits as")
        say("  independent observations, which inflates its sample size and makes")
        say("  it prefer a larger model than held-out patients support. The")
        say("  predictive criterion is the one used.")
    say("")
    say("  Severson et al. (2021) reported eight states. That is a result about")
    say("  their cohort, variables and model, not a target for ours: the count")
    say("  here is whatever the development patients support.")

    # ------------------------------------------- a fit on all development data
    section("4. REFERENCE FIT AT THE CHOSEN STATE COUNT")
    std = cs.Standardiser.fit(dev_sequences)
    fit, records = cs.fit_em_multistart(
        std.apply(dev_sequences), chosen, seed=cfg.seed, n_starts=N_STARTS)
    fit = cs.reorder(fit, cs.canonical_order(fit, std))
    fit.label_emission = cs.fit_label_emission(fit, std.apply(dev_sequences))

    starts = pd.DataFrame(records)
    starts.round(4).to_csv(out / "reference_fit_starts.csv", index=False)
    say("  EM starts (development patients, chosen state count):")
    say(block(starts.round(3), index=False))

    described = cs.describe_states(fit, std.apply(dev_sequences), std)
    for key, frame in described.items():
        frame.round(4).to_csv(out / f"reference_{key}.csv", index=False)
    say("")
    say("  State profiles, in original clinical units (development patients):")
    cols = ["state"] + list(cs.FEATURE_NAMES) + ["share_visits_hard",
                                                 "mean_sojourn_years"]
    say(block(described["profile"][cols].round(2), index=False))
    say("")
    say("  State assignment uncertainty:")
    say(block(described["uncertainty"].round(4), index=False))
    say("")
    say("  This fit is a DESCRIPTION of the development patients, not the model")
    say("  evaluated anywhere. Script 62 refits from scratch inside every LOPO")
    say("  fold on the evaluation patients.")

    # ---------------------------------------------------------- provenance
    section("5. PROVENANCE")
    env = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(),
        "node": platform.node(), "workers": args.workers, "seed": cfg.seed,
        "panel_md5": panel_md5,
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
                        [Path(__file__), ROOT / "src/ppmi_subtype/clinical_states.py"]},
        "split": {"development_share": DEVELOPMENT_SHARE,
                  "n_development": len(dev), "n_evaluation": len(evaluation),
                  "seed": cfg.seed},
        "state_counts_tested": list(STATE_COUNTS),
        "chosen_n_states": chosen,
        "selection_rule": ("smallest state count within one inner-fold standard "
                           "error of the best mean held-out predictive "
                           "log-density" if USE_ONE_SE_RULE else "arithmetic max"),
        "arithmetic_max_would_choose": arg_max_states,
        "bic_would_choose": bic_pick,
        "state_count_summary": summary.to_dict(orient="records"),
        "n_failed_cells": int(len(failed)),
        "feature_inventory": inventory.to_dict(orient="records"),
        "reference_profile": described["profile"].to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    print(f"\nWrote {out}\nCHOSEN_N_STATES={chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
