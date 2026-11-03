"""Rich clinical profiles: predict the next visit's 9-domain measurements.

The experiment
--------------
Represent each patient, in each of nine clinical domains, by a *level* and a
*rate of change* filtered from their own history, with levodopa dose entered as
an input rather than as a domain. Predict the next visit's measurements.

The primary target is the **next visit's multidimensional clinical profile** --
measured quantities, not a model-generated state identifier. The secondary
endpoint is the next TD/Indeterminate/PIGD label, derived from the predicted
tremor and PIGD scores by this project's own unmodified classification rule,
and reported separately because the subtype is a thresholded ratio of two of
the nine predicted numbers.

Design, hypotheses and pre-registered failure conditions:
``docs/latent_clinical_states/02_RICH_PROFILE_DESIGN_20261020.md``.

Validation
----------
Repeated patient-level K-fold cross-validation. Every patient is held out
exactly once per repeat, so every patient contributes held-out predictions, and
the repeats give a direct estimate of stability across splits -- one of the
evaluation axes the design asks for. Leave-one-patient-out was measured at
~153s per fit for the full model, so 934 folds times four model variants would
cost roughly eight hours for an estimate that K-fold delivers in minutes; the
choice is recorded rather than implied.

The standardiser, the model parameters, the population-mean baseline and the
LOCF spread are all refitted inside every fold on training patients only.

    python scripts/64_clinical_profile_cv.py --out <dir> [--workers N]
                                             [--folds K] [--repeats R]
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

from ppmi_subtype import clinical_profile as cp  # noqa: E402
from ppmi_subtype import clinical_states as cs  # noqa: E402
from ppmi_subtype import next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

#: The four fitted variants. The first is the primary model; the rest isolate
#: one component each so the primary model's gain can be attributed.
VARIANTS = {
    cp.MODEL_FULL: {"use_rate": True, "use_medication": True},
    cp.MODEL_NO_RATE: {"use_rate": False, "use_medication": True},
    cp.MODEL_NO_MED: {"use_rate": True, "use_medication": False},
}
ALL_MODELS = (cp.MODEL_FULL, cp.MODEL_NO_RATE, cp.MODEL_NO_MED,
              cp.MODEL_LOCF, cp.MODEL_POPMEAN)

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


def slug(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ---------------------------------------------------------------------------
#  One fold
# ---------------------------------------------------------------------------
def _init(sequences) -> None:
    _WORK["sequences"] = sequences


def run_fold(sequences, test_patients, repeat: int, fold: int):
    """Fit every variant on the training patients, predict the held-out ones."""
    test_set = set(test_patients)
    train = [s for s in sequences if s["PATNO"] not in test_set]
    test = [s for s in sequences if s["PATNO"] in test_set]
    if not train or not test:
        raise ValueError(f"empty split (repeat {repeat}, fold {fold})")
    if test_set & {s["PATNO"] for s in train}:
        raise RuntimeError("held-out patient leaked into training")

    # Every learned transformation is fitted here, on training patients only.
    standardiser = cp.Standardiser.fit(train)
    train_z = standardiser.apply(train)
    test_z = standardiser.apply(test)
    population = cp.PopulationMean.fit(train_z)
    locf_spread = cp.locf_sd(train_z)
    # Training mean per domain, for the LOCF baseline's never-recorded case.
    # On the standardised scale this is zero by construction, but it is passed
    # explicitly rather than assumed.
    locf_fallback = np.zeros(len(cp.DOMAINS))

    fits, diagnostics = {}, {}
    for name, kw in VARIANTS.items():
        m = cp.fit(train_z, **kw)
        fits[name] = m
        diagnostics[name] = {
            "train_loglik": m.loglik, "scipy_success": m.scipy_success,
            "proj_grad": m.proj_grad, "converged": m.converged,
            "n_iter": m.n_iter, "termination": m.termination,
            "n_train_patients": m.n_train_patients,
        }

    records = []
    for seq in test_z:
        # The three fitted variants each need their own filter pass.
        for name in VARIANTS:
            records += cp.predict_sequence(
                fits[name], seq, standardiser, models=(name,))
        # The two deterministic baselines share a filter pass they do not use.
        records += cp.predict_sequence(
            fits[cp.MODEL_FULL], seq, standardiser,
            population=population, locf_spread=locf_spread,
            locf_fallback=locf_fallback,
            models=(cp.MODEL_LOCF, cp.MODEL_POPMEAN))

    frame = cp.records_to_frame(records)
    frame["repeat"] = repeat
    frame["fold"] = fold
    for name, d in diagnostics.items():
        for k, v in d.items():
            frame.loc[frame["model"] == name, f"fit_{k}"] = v

    # The representation of the held-out patients, from the primary model.
    rep = cp.representation_frame(fits[cp.MODEL_FULL], test_z, standardiser)
    rep["repeat"] = repeat
    rep["fold"] = fold

    coefs = fits[cp.MODEL_FULL].coefficient_frame()
    coefs["repeat"] = repeat
    coefs["fold"] = fold
    return frame, rep, coefs


def _fold_task(args):
    repeat, fold, test_patients = args
    try:
        return run_fold(_WORK["sequences"], test_patients, repeat, fold), None
    except Exception as exc:                        # noqa: BLE001 -- recorded
        return None, {"repeat": repeat, "fold": fold,
                      "error": f"{type(exc).__name__}: {exc}"}


def make_splits(patients, n_folds: int, n_repeats: int, seed: int):
    """Patient-level folds: each patient held out exactly once per repeat."""
    patients = np.array(sorted(patients), dtype=object)
    tasks = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + 1000 * r)
        order = rng.permutation(len(patients))
        for f in range(n_folds):
            idx = order[f::n_folds]
            tasks.append((r, f, patients[idx].tolist()))
    return tasks


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--limit-patients", type=int, default=None,
                    help="smoke test on a subset of patients")
    ap.add_argument("--baseline", default=None,
                    help="subtype-history run dir, for a paired label comparison")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    # ------------------------------------------------------------------ data
    section("1. DATA AND REPRESENTATION")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel md5            {panel_md5}  [MATCH]")

    panel = pd.read_csv(panel_path, low_memory=False)
    clinical = cs.build_clinical_panel(cfg, panel)
    sequences = cp.build_sequences(clinical)
    if args.limit_patients:
        sequences = sequences[: args.limit_patients]

    say(f"  clinical visits      {len(clinical):,}")
    say(f"  patients with >= 2   {len(sequences):,}")
    say(f"  prediction cases     {sum(len(s['gaps']) for s in sequences):,}")
    say(f"  domains predicted    {len(cp.DOMAINS)}  {list(cp.DOMAINS)}")
    say(f"  medication input     {cp.MEDICATION!r} (NOT a predicted domain)")
    say("")
    say("  The representation is each patient's filtered (level, rate) per")
    say("  domain -- 18 continuous numbers with uncertainty attached, not a")
    say("  discrete state. Missing measurements are marginalised, not imputed.")

    inventory = cs.feature_inventory(clinical)
    inventory = inventory[inventory["feature"].isin(cp.DOMAINS + (cp.MEDICATION,))]
    inventory.round(4).to_csv(out / "feature_inventory.csv", index=False)
    say("")
    say(block(inventory[["feature", "domain", "coverage", "mean", "sd",
                         "min", "max"]].round(3), index=False))

    # ---------------------------------------------------------- the target
    section("2. THE PREDICTION TARGET")
    say("  PRIMARY: the next visit's 9-domain clinical profile, as a")
    say("  probability distribution, scored against the measurements actually")
    say("  recorded. A measured target, so it does not move when the model does.")
    say("")
    say("  NOT the target: the next latent state. A learned state is a model")
    say("  artifact; predicting it would show internal consistency and nothing")
    say("  about the patient.")
    say("")
    say("  SECONDARY: the next TD/Indeterminate/PIGD label, derived from the")
    say("  predicted tremor and PIGD scores by this project's own")
    say("  subtype.classify -- verified to reproduce the recorded panel label")
    say("  on 4,433 of 4,433 visits -- so no new classifier or threshold is")
    say("  introduced. Reported separately: the subtype is a thresholded ratio")
    say("  of two of the nine predicted numbers, so it tests two coordinates.")

    # ------------------------------------------------------------------- cv
    section("3. REPEATED PATIENT-LEVEL CROSS-VALIDATION")
    tasks = make_splits([s["PATNO"] for s in sequences], args.folds,
                        args.repeats, cfg.seed)
    say(f"  folds per repeat     {args.folds}")
    say(f"  repeats              {args.repeats}")
    say(f"  total fits           {len(tasks) * len(VARIANTS):,}"
        f"  ({len(tasks)} folds x {len(VARIANTS)} fitted variants)")
    say(f"  workers              {args.workers}")
    say("  Refitted per fold: standardiser, model parameters, population-mean")
    say("  baseline, LOCF spread. Nothing is fitted on a held-out patient.")

    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers, initializer=_init,
                                         initargs=(sequences,)) as pool:
            results = []
            for n, r in enumerate(pool.imap(_fold_task, tasks), start=1):
                results.append(r)
                say(f"  fold {n}/{len(tasks)} done")
    else:
        _init(sequences)
        results = [_fold_task(t) for t in tasks]

    frames, reps, coefs, failures = [], [], [], []
    for (res, err) in results:
        if err is not None:
            failures.append(err)
            continue
        f, rep, c = res
        frames.append(f)
        reps.append(rep)
        coefs.append(c)

    if not frames:
        raise SystemExit("every fold failed")

    preds = pd.concat(frames, ignore_index=True)
    representation = pd.concat(reps, ignore_index=True)
    coefficients = pd.concat(coefs, ignore_index=True)
    fails = pd.DataFrame(failures, columns=["repeat", "fold", "error"])
    fails.to_csv(out / "failed_folds.csv", index=False)

    say("")
    say(f"  folds completed      {len(frames)}/{len(tasks)}")
    say(f"  folds failed         {len(fails)}")
    if len(fails):
        say(block(fails, index=False))
    say(f"  held-out predictions {len(preds):,} rows "
        f"({preds['model'].nunique()} models)")
    say(f"  elapsed              {time.time() - started:.0f}s")

    conv = coefficients  # per-fold primary-model parameters
    fold_diag = (preds[preds["model"].isin(VARIANTS)]
                 .drop_duplicates(["model", "repeat", "fold"]))
    say("")
    say("  Convergence of the fitted variants:")
    cdiag = (fold_diag.groupby("model")
             .agg(n_folds=("fit_converged", "size"),
                  n_converged=("fit_converged", "sum"),
                  n_scipy_success=("fit_scipy_success", "sum"),
                  max_proj_grad=("fit_proj_grad", "max"),
                  mean_iter=("fit_n_iter", "mean"))
             .reset_index())
    cdiag.round(4).to_csv(out / "convergence.csv", index=False)
    say(block(cdiag.round(3), index=False))
    say("")
    say("  scipy's success flag is not the criterion on its own: with several")
    say("  variances at their lower bound, L-BFGS-B stops on a boundary")
    say("  condition where the likelihood has already stopped moving. The")
    say("  projected gradient relative to the log-likelihood scale is used, and")
    say("  both are reported.")

    # Primary analysis uses repeat 0 so each prediction appears once; the other
    # repeats are the stability estimate.
    primary = preds[preds["repeat"] == 0].reset_index(drop=True)
    preds.to_csv(out / "profile_predictions.csv", index=False)
    representation.to_csv(out / "representation.csv", index=False)
    coefficients.round(6).to_csv(out / "fold_coefficients.csv", index=False)

    # ------------------------------------------------- H1 prediction quality
    section("4. H1: PROFILE PREDICTION QUALITY (primary target)")
    rows = [cp.profile_metrics(g.reset_index(drop=True), seed=cfg.seed)
            for _, g in primary.groupby("model", sort=False)]
    comparison = pd.DataFrame([{k: v for k, v in r.items() if k != "per_domain"}
                               for r in rows])
    comparison = comparison.sort_values("loglik_per_domain_observation",
                                        ascending=False)
    comparison.round(4).to_csv(out / "model_comparison.csv", index=False)
    show = ["model", "n_predictions", "loglik_per_domain_observation",
            "loglik_ci_low", "loglik_ci_high", "rmse_standardised",
            "mae_standardised", "coverage_95"]
    say(block(comparison[show].round(4), index=False))
    say("")
    n_fb = int(primary.loc[primary["model"] == cp.MODEL_LOCF,
                           "n_domains_fallback"].sum())
    say("")
    say(f"  Last-observation-carried-forward had no value to carry in "
        f"{n_fb:,} domain-cases")
    say("  (a domain never recorded in that patient's history up to the")
    say("  prediction point) and fell back to the TRAINING mean for those.")
    say("")
    say("  loglik_per_domain_observation is the Gaussian predictive log-density")
    say("  of the next visit's measurements, per domain actually recorded;")
    say("  higher is better. rmse_standardised is on the standardised scale so")
    say("  domains are commensurable. coverage_95 should be near 0.95.")

    per_domain = []
    for r in rows:
        for d in r.get("per_domain", []):
            per_domain.append({"model": r["model"], **d})
    pd.DataFrame(per_domain).round(4).to_csv(out / "per_domain_errors.csv",
                                             index=False)

    section("5. H1: PAIRED CONTRASTS")
    pairs = [("rich profile against assume-no-change", cp.MODEL_FULL, cp.MODEL_LOCF),
             ("rich profile against the population average", cp.MODEL_FULL,
              cp.MODEL_POPMEAN),
             ("modelling the rate of change", cp.MODEL_FULL, cp.MODEL_NO_RATE),
             ("medication as an input", cp.MODEL_FULL, cp.MODEL_NO_MED)]
    crows = []
    for what, a, b in pairs:
        fa = primary[primary["model"] == a]
        fb = primary[primary["model"] == b]
        if not len(fa) or not len(fb):
            continue
        crows.append({"contrast": what, "model_a": a, "model_b": b,
                      **cp.paired_loglik_difference(fa, fb, seed=cfg.seed)})
    contrasts = pd.DataFrame(crows)
    contrasts.round(4).to_csv(out / "contrasts.csv", index=False)
    say(block(contrasts[["contrast", "n_paired", "loglik_delta",
                         "delta_ci_low", "delta_ci_high",
                         "share_a_better"]].round(4), index=False))
    say("")
    say("  Differences are per-prediction mean log-density, paired on identical")
    say("  held-out visits, with patient-level bootstrap intervals.")

    # --------------------------------------------- H2 representation quality
    section("6. H2: IS THE REPRESENTATION MORE THAN ONE SEVERITY AXIS?")
    rep0 = representation[representation["repeat"] == 0]
    dim_rows = []
    for which in ("level", "rate"):
        d = cp.representation_dimensionality(rep0, which=which)
        dim_rows.append(d)
        say(f"  {which}: PC1 {d.get('pc1_share', float('nan')):.3f}  "
            f"PC1+PC2 {d.get('cumulative_2', float('nan')):.3f}  "
            f"components for 90% {d.get('n_components_for_90pct')}  "
            f"participation ratio {d.get('effective_rank_participation', float('nan')):.2f}")
    pd.DataFrame(dim_rows).to_csv(out / "representation_dimensionality.csv",
                                  index=False)
    say("")
    say("  H2 was pre-registered to FAIL if one component explains > 90% of")
    say("  the across-patient variance, which would make the representation a")
    say("  severity score with nine coordinates.")

    section("7. H2/H3: PER-DOMAIN RATES OF CHANGE")
    spread = cp.rate_spread(rep0)
    spread.round(4).to_csv(out / "rate_spread.csv", index=False)
    say(block(spread.round(4), index=False))
    say("")
    say("  Rates are in clinical units per year. A representation that captured")
    say("  only average severity would shrink these to a common value.")

    # ---------------------------------------------------- secondary endpoint
    section("8. SECONDARY ENDPOINT: THE DERIVED TD/PIGD LABEL")
    say("  Derived from the predicted tremor and PIGD distributions by this")
    say("  project's own subtype.classify, via Monte Carlo over the predictive")
    say("  Gaussians. No new classifier is fitted.")
    label_rows = []
    label_frames = []
    for name, g in primary.groupby("model", sort=False):
        lab = cp.derive_label_probabilities(g.reset_index(drop=True), cfg,
                                            n_draws=2000, seed=cfg.seed)
        probs = lab[[f"p_{s}" for s in cs.STATE_NAMES]].to_numpy(float)
        p_td, p_pigd, defined = nv.conditional_two_class(probs)
        lab["p_TD_raw"], lab["p_Indeterminate_raw"], lab["p_PIGD_raw"] = \
            probs[:, 0], probs[:, 1], probs[:, 2]
        lab["P_TD"], lab["P_PIGD"] = p_td, p_pigd
        lab["two_class_defined"] = defined
        lab["predicted_state"] = nv.predicted_state(p_td, p_pigd)
        lab["is_tie"] = defined & (p_td == p_pigd)
        lab["eligible"] = lab["actual_next_label"].isin(["TD", "PIGD"])
        label_frames.append(lab)
        label_rows.append(nv.two_class_metrics(lab, seed=cfg.seed))
    label_preds = pd.concat(label_frames, ignore_index=True)
    label_cmp = pd.DataFrame(label_rows).sort_values("accuracy", ascending=False)
    label_cmp.round(4).to_csv(out / "derived_label_comparison.csv", index=False)
    keep = ["PATNO", "current_event", "target_event", "interval_years",
            "current_label", "actual_next_label", "model", "P_TD", "P_PIGD",
            "predicted_state", "eligible", "two_class_defined",
            "p_TD_raw", "p_Indeterminate_raw", "p_PIGD_raw"]
    label_preds[keep].to_csv(out / "derived_label_predictions.csv", index=False)
    say("")
    say(block(label_cmp[["model", "n_scored", "accuracy",
                         "accuracy_ci_patient_bootstrap_low",
                         "accuracy_ci_patient_bootstrap_high",
                         "brier_two_class", "log_loss"]].round(4), index=False))
    if args.baseline:
        bl = pd.read_csv(Path(args.baseline) / "next_visit_predictions.csv",
                         low_memory=False)
        key = ["PATNO", "current_event", "target_event"]
        wanted = label_preds[key].drop_duplicates()
        bl = bl.merge(wanted, on=key, how="inner")
        keep_models = ["CT-HMM (latent, interval-aware)",
                       "Persistence (last non-Indeterminate)",
                       "Markov (observed, discrete)"]
        bl = bl[bl["model"].isin(keep_models)].copy()
        bl["model"] = "Subtype-history: " + bl["model"]

        cols = list(set(label_preds.columns) & set(bl.columns))
        joint = pd.concat([label_preds[cols], bl[cols]], ignore_index=True)

        # Restrict to the cases every model has, so the contrast is paired.
        common = None
        for _, g in joint.groupby("model", sort=False):
            ss = set(map(tuple, g[key].to_numpy()))
            common = ss if common is None else (common & ss)
        mask = np.array([tuple(r) in common for r in joint[key].to_numpy()])
        n_dropped = int((~mask).sum())
        joint = joint[mask].reset_index(drop=True)

        say("")
        say(f"  Paired against the subtype-history experiment on matched visit")
        say(f"  pairs: {len(common):,} cases shared by all models"
            + (f"; {n_dropped:,} rows dropped to pair them." if n_dropped else "."))
        jrows = [nv.two_class_metrics(g.reset_index(drop=True), seed=cfg.seed)
                 for _, g in joint.groupby("model", sort=False)]
        jcmp = pd.DataFrame(jrows).sort_values("accuracy", ascending=False)
        jcmp.round(4).to_csv(out / "derived_label_vs_subtype_history.csv",
                             index=False)
        say("")
        say(block(jcmp[["model", "n_scored", "accuracy",
                        "accuracy_ci_patient_bootstrap_low",
                        "accuracy_ci_patient_bootstrap_high",
                        "brier_two_class", "log_loss"]].round(4), index=False))

        ref = "Subtype-history: CT-HMM (latent, interval-aware)"
        jc = []
        for a in (cp.MODEL_FULL, cp.MODEL_NO_RATE):
            if a in set(joint["model"]) and ref in set(joint["model"]):
                jc.append({"contrast": f"{a} vs subtype-history CT-HMM",
                           **nv.paired_accuracy_difference(
                               joint[joint["model"] == a],
                               joint[joint["model"] == ref], seed=cfg.seed)})
        if jc:
            jcontrast = pd.DataFrame(jc)
            jcontrast.round(4).to_csv(out / "derived_label_contrasts.csv",
                                      index=False)
            say("")
            say(block(jcontrast[["contrast", "accuracy_delta",
                                 "delta_ci_patient_bootstrap_low",
                                 "delta_ci_patient_bootstrap_high",
                                 "a_right_b_wrong", "b_right_a_wrong"]].round(4),
                      index=False))
        say("")
        say("  H4 predicted in advance that the profile route would NOT beat the")
        say("  subtype-history model on this endpoint. The paired contrast above")
        say("  is the test of that prediction.")
    else:
        say("")
        say("  No --baseline supplied, so no paired comparison against the")
        say("  subtype-history experiment is reported.")

    # ------------------------------------------- H3 clinical relevance
    section("9. H3: CLINICAL RELEVANCE AGAINST INDEPENDENT OUTCOMES")
    say("  The representation is taken at each patient's SECOND visit, so it is")
    say("  formed from their first two visits only. An outcome counts only if")
    say("  recorded strictly LATER than that visit, making this prospective.")
    say("  None of these outcomes is one of the nine domains:")
    for name, spec in cp.OUTCOMES.items():
        flag = "independent" if spec["independent"] else "NOT fully independent"
        say(f"    {name:22s} {spec['label']}  [{flag}]")

    # The raw panel, not the clinical frame: the clinical frame carries only
    # the nine domains, so the outcome columns are not in it.
    missing_cols = [spec["column"] for spec in cp.OUTCOMES.values()
                    if spec["column"] not in panel.columns]
    if missing_cols:
        raise SystemExit(f"panel is missing outcome columns: {missing_cols}")
    outcome_table = cp.build_outcome_table(panel, rep0, anchor_visit=1)
    # Full precision, like the other per-patient tables this script saves. It
    # was previously rounded to 4dp, which is destructive for the rate
    # coordinates: those are of order 1e-3, so rounding collapses many of them
    # to exactly zero and perturbs any rank statistic recomputed from the file.
    # The analysis below uses the in-memory table either way, so this only
    # affects whether the saved file can be reused -- and it could not be.
    outcome_table.to_csv(out / "outcome_table.csv", index=False)
    assoc = cp.outcome_associations(outcome_table, seed=cfg.seed)
    assoc.round(4).to_csv(out / "outcome_associations.csv", index=False)

    say("")
    say(f"  patients with a second visit and later follow-up: "
        f"{len(outcome_table):,}")
    ev = (assoc.drop_duplicates("outcome")[["outcome", "n", "n_events"]]
          .reset_index(drop=True))
    say(block(ev, index=False))

    for oc in assoc["outcome"].unique():
        sub = assoc[(assoc["outcome"] == oc) & assoc["auc"].notna()]
        if not len(sub):
            say(f"\n  {oc}: skipped (too few events)")
            continue
        top = sub.sort_values("auc_abs", ascending=False).head(6)
        say("")
        say(f"  {oc} -- strongest univariate discriminators:")
        say(block(top[["score", "auc", "auc_abs"]].round(4), index=False))
        ref_row = sub[sub["score"] == "subtype label (reference)"]
        if len(ref_row):
            say(f"    subtype label (reference) AUC "
                f"{float(ref_row['auc'].iloc[0]):.4f}")

    h3 = []
    for oc in assoc["outcome"].unique():
        if assoc[(assoc["outcome"] == oc)]["auc"].notna().any():
            for kind in ("rate", "level"):
                h3.append(cp.best_component_vs_subtype(
                    outcome_table, assoc, oc, kind=kind, seed=cfg.seed))
    h3f = pd.DataFrame([r for r in h3 if r.get("n")])
    if len(h3f):
        h3f.round(4).to_csv(out / "h3_component_vs_subtype.csv", index=False)
        say("")
        say("  Best single component minus the subtype label (AUC), with a")
        say("  patient bootstrap. The component is chosen on the same data it is")
        say("  compared on, so these are UPPER BOUNDS on what one coordinate buys:")
        say(block(h3f[["outcome", "kind", "best_component", "auc_a", "auc_b",
                       "delta", "delta_ci_low", "delta_ci_high"]].round(4),
                  index=False))

    # ---------------------------------------------------------- robustness
    section("10. ROBUSTNESS ACROSS SPLITS")
    rob = []
    for (rep_i, model), g in preds.groupby(["repeat", "model"], sort=False):
        m = cp.profile_metrics(g.reset_index(drop=True), seed=cfg.seed,
                               n_boot=200)
        rob.append({"repeat": rep_i, "model": model,
                    "loglik_per_domain_observation":
                        m["loglik_per_domain_observation"],
                    "rmse_standardised": m["rmse_standardised"]})
    robust = pd.DataFrame(rob)
    robust.round(4).to_csv(out / "robustness_by_repeat.csv", index=False)
    pivot = robust.pivot(index="model", columns="repeat",
                         values="loglik_per_domain_observation")
    pivot["sd_across_repeats"] = pivot.std(axis=1)
    say(block(pivot.round(4)))
    say("")
    say("  Parameter stability across folds (primary model):")
    pstab = (coefficients.groupby("domain")
             .agg(obs_var_mean=("obs_var", "mean"), obs_var_sd=("obs_var", "std"),
                  level_var_mean=("level_innovation_var", "mean"),
                  level_var_sd=("level_innovation_var", "std"),
                  rate_var_mean=("rate_innovation_var", "mean"),
                  rate_var_sd=("rate_innovation_var", "std"),
                  beta_mean=("medication_beta", "mean"),
                  beta_sd=("medication_beta", "std"))
             .reindex(list(cp.DOMAINS)).reset_index())
    pstab.round(4).to_csv(out / "parameter_stability.csv", index=False)
    say(block(pstab.round(4), index=False))

    # ---------------------------------------------------------- checks
    section("11. CORRECTNESS CHECKS")
    model_counts = primary.groupby("model").size()
    checks = {
        "every model scored the identical held-out cases":
            model_counts.nunique() == 1,
        "history length is the visits up to the current one":
            bool((primary["n_history"] == primary["visit_index"] + 1).all()),
        "target visit is strictly later than the current visit":
            bool((primary["target_years"] > primary["current_years"]).all()),
        "elapsed time equals the gap between visit times":
            bool(np.allclose(primary["interval_years"],
                             primary["target_years"] - primary["current_years"],
                             atol=1e-8)),
        "predictive standard deviations are finite and positive":
            bool(all(np.isfinite(primary[f"sd_{d}"]).all()
                     and (primary[f"sd_{d}"] > 0).all() for d in cp.DOMAINS)),
        "predicted means are finite":
            bool(all(np.isfinite(primary[f"pred_{d}"]).all() for d in cp.DOMAINS)),
        "medication is not a predicted domain":
            cp.MEDICATION not in cp.DOMAINS,
        "every patient appears in exactly one fold per repeat":
            bool((preds.groupby(["repeat", "PATNO"])["fold"].nunique() == 1).all()),
        "every patient is held out in every repeat":
            bool((preds.groupby("PATNO")["repeat"].nunique()
                  == preds["repeat"].nunique()).all()),
        "each fold trained on the other patients only":
            bool((fold_diag["fit_n_train_patients"]
                  < len(sequences)).all()),
        "derived label probabilities sum to one":
            bool(np.allclose(label_preds[["p_TD_raw", "p_Indeterminate_raw",
                                          "p_PIGD_raw"]].sum(axis=1), 1.0,
                             atol=1e-9)),
        "two-class probabilities sum to one where defined":
            bool(np.allclose(
                label_preds.loc[label_preds["two_class_defined"], "P_TD"]
                + label_preds.loc[label_preds["two_class_defined"], "P_PIGD"],
                1.0, atol=1e-12)),
        "no fold failed": len(fails) == 0,
    }
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # ---------------------------------------------------------- provenance
    section("12. PROVENANCE")
    env = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(),
        "node": platform.node(), "workers": args.workers, "seed": cfg.seed,
        "folds": args.folds, "repeats": args.repeats,
        "domains": list(cp.DOMAINS), "medication_input": cp.MEDICATION,
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
                        [Path(__file__), ROOT / "src/ppmi_subtype/clinical_profile.py",
                         ROOT / "src/ppmi_subtype/clinical_states.py",
                         ROOT / "src/ppmi_subtype/next_visit.py"]},
        "counts": {"n_patients": len(sequences),
                   "n_cases": int(len(primary) / primary["model"].nunique()),
                   "n_folds": len(tasks), "n_failed": int(len(fails))},
        "checks": checks,
        "model_comparison": comparison.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "representation_dimensionality": dim_rows,
        "rate_spread": spread.to_dict(orient="records"),
        "derived_label_comparison": label_cmp.to_dict(orient="records"),
        "robustness": robust.to_dict(orient="records"),
        "parameter_stability": pstab.to_dict(orient="records"),
        "outcome_associations": assoc.to_dict(orient="records"),
        "h3_component_vs_subtype": (h3f.to_dict(orient="records")
                                    if len(h3f) else []),
    }, indent=2, default=float), encoding="utf-8")
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
