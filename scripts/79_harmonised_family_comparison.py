"""E-8: one decision rule, one baseline, one metric suite across every model family.

THE PROBLEM
-----------
Nine model specifications spanning five families were all scored on the same
3,361 held-out transitions from the same 934 patients. They were **not** scored
under the same label-derivation rule, and the rule is worth more than the
differences between the families:

    gp_indep_K1   pred_subtype 0.7298   prob_subtype 0.7566   (+0.0268)
    persistence   pred_subtype 0.7227   prob_subtype 0.7432   (+0.0205)

Same model, same saved predictions. So the project's headline margin over
persistence is measured against a different baseline for different families,
and no claim of the form "model X beats the trivial rule by Y" is currently
well defined across the comparison. This script builds the table that is.

WHAT IS HARMONISED
------------------
*   **One representation.** Every model is expressed as a 3-class predictive
    probability vector over (TD, Indeterminate, PIGD) on the same cases. The
    chains and HMMs already are; the subscore models are converted by Monte
    Carlo through the project's own classification rule.
*   **One decision rule.** The predicted class is the argmax of that vector.
*   **One canonical baseline.** Label persistence -- name the currently
    recorded label. It is the trivial rule a clinician would use and needs no
    subscores. The subscore-LOCF baseline is reported too, clearly separated,
    because it is a different (model-ish) construction and conflating the two
    is how the comparison went wrong in the first place.
*   **One metric suite and one bootstrap.** Accuracy, multiclass Brier, log
    loss, per-class recall and macro-F1, with a patient-clustered bootstrap and
    paired contrasts against the canonical baseline.

TWO DECISIONS THAT CHANGE PUBLISHED NUMBERS, BOTH DELIBERATE
------------------------------------------------------------
1.  **Both-zero draws go to Indeterminate, not TD.** `scripts/18` computes the
    draw ratio as `np.divide(..., out=inf, where=draw_pigd > 0)`, so a draw
    with both subscores zero gets `ratio = inf` and is classified TD. The
    project's own `subtype.classify` sends tremor == 0 and pigd == 0 to
    **Indeterminate**. 0.59% of draws are affected, touching 2,230 of 3,361
    cases and changing the argmax label on 14. This script follows
    `subtype.classify`, and an equivalence test against the real function is a
    validation check rather than an assertion.
2.  **A fresh Monte Carlo, with common random numbers.** The saved draws cannot
    be reproduced bit-for-bit because the RNG stream differs, so this is a new
    Monte Carlo, not a re-derivation. To stop that noise contaminating the
    paired contrasts, **every model sees the identical standard-normal draws
    for the identical case** (seeded per chunk, not per model), so Monte Carlo
    error is shared and largely cancels in differences. Residual error is
    estimated and reported.

    python scripts/79_harmonised_family_comparison.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import subtype as st  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK, INK_MUTED, _finish, _style  # noqa: E402

CLASSES = ["TD", "Indeterminate", "PIGD"]
PCOLS = ["p_TD", "p_Indeterminate", "p_PIGD"]
N_CASES = 3361
CANONICAL = "Persistence (recorded label)"

#  (source file, model name in that file, display name, family).
#  Baselines that appear in several files are taken once, from one source.
LABEL_SOURCES = [
    ("outputs/loo/loo_predictions.csv", "HMM",
     "CT-HMM, 3 latent states", "latent-state HMM"),
    ("outputs/loo_5hidden/loo_predictions_all_models.csv", "HMM",
     "CT-HMM, 5 latent states", "latent-state HMM"),
    ("outputs/loo/loo_predictions.csv", "Markov (discrete)",
     "Markov chain, discrete", "observed-label chain"),
    ("outputs/loo/loo_predictions.csv", "CTMC (interval-aware)",
     "CTMC, interval-aware", "observed-label chain"),
    ("outputs/loo/loo_predictions.csv", "Persistence (stay)",
     CANONICAL, "baseline"),
    ("outputs/loo/loo_predictions.csv", "Marginal (null)",
     "Marginal null", "baseline"),
]
SUBSCORE_SOURCES = [
    ("outputs/joint_subscore_gp/predictions_with_subtype.csv", "gp_indep_K1",
     "Subscore GP, independent", "subscore GP"),
    ("outputs/joint_subscore_gp/predictions_with_subtype.csv", "gp_joint_K1_full",
     "Subscore GP, joint", "subscore GP"),
    ("outputs/joint_subscore/predictions_with_subtype.csv", "joint_lmm",
     "Joint subscore LMM", "subscore LMM"),
    ("outputs/time_varying/predictions_with_subtype.csv", "lmm_spline_2_5",
     "Time-varying LMM, spline", "time-varying LMM"),
    ("outputs/time_varying/predictions_with_subtype.csv", "lmm_constant",
     "Time-varying LMM, constant", "time-varying LMM"),
    ("outputs/joint_subscore_gp/predictions_with_subtype.csv", "persistence",
     "Persistence (subscore LOCF)", "baseline"),
    ("outputs/joint_subscore_gp/predictions_with_subtype.csv", "patient_mean",
     "Patient mean subscore", "baseline"),
    ("outputs/joint_subscore/predictions_with_subtype.csv", "population",
     "Population mean subscore", "baseline"),
]

ACCENT = "#1f4e79"
WARM = "#c2552c"
MUTED = "#8c8c88"

lines: list[str] = []


def say(t: str = "") -> None:
    print(t, flush=True)
    lines.append(t)


def section(t: str) -> None:
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(df: pd.DataFrame, index: bool = False) -> str:
    return "    " + df.to_string(index=index).replace("\n", "\n    ")


# ---------------------------------------------------------------------------
#  Classification of Monte Carlo draws, following subtype.classify exactly
# ---------------------------------------------------------------------------
def classify_draws(tremor: np.ndarray, pigd: np.ndarray, cfg) -> tuple:
    """Vectorised `subtype.classify` over an array of draws.

    The degenerate cases are the whole point and are written out rather than
    left to a division:

        pigd == 0 and tremor >  0  -> TD
        tremor == 0 and pigd == 0  -> Indeterminate
        otherwise, ratio against the two thresholds

    `test_matches_subtype_classify` checks this against the real function.
    """
    sc = cfg.subtype
    td_thr, pigd_thr = sc.td_threshold, sc.pigd_threshold

    divisible = pigd > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(divisible, tremor / np.where(divisible, pigd, 1.0), np.nan)

    if sc.band_boundaries == "inner":
        is_td = divisible & (ratio > td_thr)
        is_pigd = divisible & (ratio < pigd_thr)
    else:
        is_td = divisible & (ratio >= td_thr)
        is_pigd = divisible & (ratio <= pigd_thr)

    pigd_zero_only = (~divisible) & (tremor > 0)
    is_td = is_td | pigd_zero_only
    #  both zero falls through to Indeterminate, as classify does
    return is_td, is_pigd


def derive_probabilities(df: pd.DataFrame, cfg, n_draws: int,
                         chunk: int = 400) -> np.ndarray:
    """3-class probabilities from a bivariate Gaussian predictive, by Monte Carlo.

    Common random numbers: the RNG is seeded from (seed, chunk index) only, so
    the same case receives the same standard-normal draws under every model and
    the Monte Carlo error cancels in paired contrasts.
    """
    mt = df["pred_tremor"].to_numpy(float)
    mp = df["pred_pigd"].to_numpy(float)
    s_t = df["sd_tremor"].to_numpy(float)
    s_p = df["sd_pigd"].to_numpy(float)
    if "corr_pred" in df.columns:
        rho = np.clip(np.nan_to_num(df["corr_pred"].to_numpy(float)), -0.999, 0.999)
    else:
        #  The LMM and time-varying files carry no cross-outcome correlation;
        #  their own reports put it near zero, so independence is used and
        #  recorded as an assumption rather than hidden.
        rho = np.zeros(len(df))

    out = np.empty((len(df), 3))
    for c0 in range(0, len(df), chunk):
        c1 = min(c0 + chunk, len(df))
        rng = np.random.default_rng([cfg.seed, c0 // chunk])
        k = c1 - c0
        z1 = rng.standard_normal((k, n_draws))
        z2 = rng.standard_normal((k, n_draws))
        r = rho[c0:c1, None]
        dt = np.clip(mt[c0:c1, None] + s_t[c0:c1, None] * z1, 0.0, None)
        dp = np.clip(mp[c0:c1, None] + s_p[c0:c1, None]
                     * (r * z1 + np.sqrt(1.0 - r ** 2) * z2), 0.0, None)
        is_td, is_pigd = classify_draws(dt, dp, cfg)
        p_td = is_td.mean(axis=1)
        p_pigd = is_pigd.mean(axis=1)
        out[c0:c1, 0] = p_td
        out[c0:c1, 2] = p_pigd
        out[c0:c1, 1] = np.clip(1.0 - p_td - p_pigd, 0.0, 1.0)
    #  renormalise against floating-point drift
    return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def metrics(P: np.ndarray, y: np.ndarray) -> dict:
    """P is (n, 3) probabilities; y is (n,) integer class index."""
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(((P - onehot) ** 2).sum(axis=1).mean())
    ll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())

    out = {"accuracy": correct.mean(), "brier_multiclass": brier, "log_loss": ll}
    f1s = []
    for k, name in enumerate(CLASSES):
        tp = float(((pred == k) & (y == k)).sum())
        fp = float(((pred == k) & (y != k)).sum())
        fn = float(((pred != k) & (y == k)).sum())
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0
              else 0.0)
        out[f"recall_{name}"] = rec
        out[f"n_predicted_{name}"] = float((pred == k).sum())
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def patient_bootstrap_mean(values: np.ndarray, patients: np.ndarray,
                           n_boot: int, seed: int) -> np.ndarray:
    """Cluster-bootstrap draws of the mean of `values`, resampling patients.

    The mean over a patient resample is the ratio of summed per-patient totals
    to summed per-patient counts, so each replicate costs O(n_patients) rather
    than rebuilding the case-level vector. That is exact, not an approximation.
    """
    codes, _ = pd.factorize(patients, sort=True)
    n_pat = codes.max() + 1
    sums = np.bincount(codes, weights=values, minlength=n_pat)
    counts = np.bincount(codes, minlength=n_pat).astype(float)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_pat, n_pat)
        draws[b] = sums[pick].sum() / counts[pick].sum()
    return draws


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=10000,
                    help="Monte Carlo draws per case for the subscore models")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = load_config()
    checks: dict[str, bool] = {}

    section("E-8  HARMONISED MULTI-FAMILY COMPARISON")
    say(f"  Monte Carlo draws per case : {args.n_draws:,}")
    say(f"  bootstrap draws            : {args.n_boot:,}   (seed {cfg.seed})")
    say(f"  canonical baseline         : {CANONICAL}")

    # ------------------------------------------------- vectorised rule check
    section("1. THE CLASSIFICATION RULE, CHECKED AGAINST subtype.classify")
    rng = np.random.default_rng(0)
    tt = np.concatenate([rng.uniform(0, 3, 4000), np.zeros(400), rng.uniform(0, 0.3, 600)])
    pp = np.concatenate([rng.uniform(0, 3, 4000), np.zeros(400), np.zeros(600)])
    ref = st.classify(pd.Series(tt), pd.Series(pp), cfg.subtype).map(cfg.subtype.label_names)
    mine_td, mine_pigd = classify_draws(tt, pp, cfg)
    mine = np.where(mine_td, "TD", np.where(mine_pigd, "PIGD", "Indeterminate"))
    agree = int((mine == ref.to_numpy()).sum())
    say(f"  {agree:,} of {len(tt):,} agree "
        f"(includes {int(((tt==0)&(pp==0)).sum())} both-zero and "
        f"{int(((pp==0)&(tt>0)).sum())} pigd-zero cases)")
    checks["vectorised rule matches subtype.classify on every test case"] = agree == len(tt)

    # ----------------------------------------------------------- label models
    section("2. LOADING MODELS ALREADY ON A 3-CLASS PROBABILITY VECTOR")
    names = cfg.subtype.label_names
    frames: dict[str, pd.DataFrame] = {}
    meta: list[dict] = []

    for path, model, disp, fam in LABEL_SOURCES:
        d = pd.read_csv(ROOT / path, low_memory=False)
        d = d[d.model == model].copy()
        d["n_history"] = d["visit_index"] + 1
        d["actual"] = d["to_state"].map(names)
        g = d[["PATNO", "n_history", "actual"] + PCOLS].rename(columns={"PATNO": "patient"})
        frames[disp] = g.sort_values(["patient", "n_history"]).reset_index(drop=True)
        meta.append({"model": disp, "family": fam, "source": path,
                     "source_model": model, "derivation": "native 3-class posterior"})
        say(f"  {disp:<32s} {len(g):>6,} rows   {path}")

    # -------------------------------------------------------- subscore models
    section("3. DERIVING 3-CLASS PROBABILITIES FOR THE SUBSCORE MODELS")
    say("  Monte Carlo through the project's own classification rule, with")
    say("  common random numbers so the draws cancel in paired contrasts.\n")
    for path, model, disp, fam in SUBSCORE_SOURCES:
        d = pd.read_csv(ROOT / path, low_memory=False)
        d = d[d.model == model].copy()
        d = d.sort_values(["held_out_patient", "history_length"]).reset_index(drop=True)
        has_rho = "corr_pred" in d.columns
        P = derive_probabilities(d, cfg, args.n_draws)
        g = pd.DataFrame({"patient": d["held_out_patient"].to_numpy(),
                          "n_history": d["history_length"].to_numpy(),
                          "actual": d["obs_subtype"].to_numpy()})
        g[PCOLS] = P
        frames[disp] = g
        meta.append({"model": disp, "family": fam, "source": path,
                     "source_model": model,
                     "derivation": f"Monte Carlo, {args.n_draws:,} draws, "
                                   f"{'joint covariance' if has_rho else 'independent (no corr column)'}"})
        say(f"  {disp:<32s} {len(g):>6,} rows   "
            f"{'joint cov' if has_rho else 'independent'}   {path}")

    # --------------------------------------------------------------- the join
    section("4. JOINING EVERY MODEL ONTO ONE CASE LIST")
    base = frames[CANONICAL][["patient", "n_history", "actual"]].copy()
    say(f"  case list from the canonical baseline: {len(base):,} cases, "
        f"{base.patient.nunique():,} patients")
    checks["the case list has the expected 3,361 transitions"] = len(base) == N_CASES

    all_ok, outcome_ok = True, True
    for disp, g in frames.items():
        m = base.merge(g, on=["patient", "n_history"], how="left", suffixes=("", "_m"))
        if m[PCOLS].isna().any().any() or len(m) != len(base):
            all_ok = False
            say(f"  INCOMPLETE JOIN for {disp}")
        if not (m["actual"] == m["actual_m"]).all():
            outcome_ok = False
            n_bad = int((m["actual"] != m["actual_m"]).sum())
            say(f"  OUTCOME MISMATCH for {disp}: {n_bad} cases")
        base[[f"{disp}|{c}" for c in PCOLS]] = m[PCOLS].to_numpy()
    say(f"  every model joined completely:            {all_ok}")
    say(f"  observed outcome identical across sources: {outcome_ok}")
    checks["every model joins completely onto the case list"] = all_ok
    checks["the observed outcome is identical across all source files"] = outcome_ok

    y = pd.Categorical(base["actual"], categories=CLASSES).codes.astype(int)
    checks["every case has a valid observed class"] = bool((y >= 0).all())
    say(f"  outcome mix: " + ", ".join(
        f"{c} {int((y == k).sum()):,}" for k, c in enumerate(CLASSES)))

    probs = {d: base[[f"{d}|{c}" for c in PCOLS]].to_numpy() for d in frames}
    finite = all(np.isfinite(P).all() and (P >= -1e-12).all()
                 and np.abs(P.sum(1) - 1).max() < 1e-9 for P in probs.values())
    checks["all probability vectors are finite, non-negative and sum to one"] = finite
    say(f"  probability vectors valid: {finite}")

    # ------------------------------------------------------------- the table
    section("5. THE HARMONISED TABLE")
    say("  One decision rule (argmax), one case list, one metric suite.\n")
    fam_of = {m["model"]: m["family"] for m in meta}
    rows = []
    for disp, P in probs.items():
        rows.append({"model": disp, "family": fam_of[disp], **metrics(P, y)})
    tab = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    tab.to_csv(out / "harmonised_metrics.csv", index=False)
    show = tab[["model", "family", "accuracy", "brier_multiclass", "log_loss",
                "macro_f1", "recall_TD", "recall_Indeterminate", "recall_PIGD",
                "n_predicted_Indeterminate"]]
    say(frame(show.round(4)))

    acc = tab.set_index("model")["accuracy"]
    non_base = tab[tab.family != "baseline"]
    say(f"\n  spread across the {len(non_base)} non-baseline specifications: "
        f"{non_base.accuracy.min():.4f} to {non_base.accuracy.max():.4f} "
        f"({100*(non_base.accuracy.max()-non_base.accuracy.min()):.1f} points)")
    say(f"  canonical baseline ({CANONICAL}): {acc[CANONICAL]:.4f}")
    say(f"  models at or below the canonical baseline: "
        f"{int((non_base.accuracy <= acc[CANONICAL]).sum())} of {len(non_base)}")
    say(f"  models predicting Indeterminate at all: "
        f"{int((tab.n_predicted_Indeterminate > 0).sum())} of {len(tab)}")

    # --------------------------------------------------------- paired contrasts
    section("6. PAIRED CONTRASTS AGAINST THE CANONICAL BASELINE")
    say("  Identical cases, patients resampled, one resample shared by both")
    say("  conditions in every replicate.\n")
    pat = base["patient"].to_numpy()
    Pb = probs[CANONICAL]
    base_correct = (Pb.argmax(1) == y).astype(float)

    rows = []
    for disp, P in probs.items():
        if disp == CANONICAL:
            continue
        d = (P.argmax(1) == y).astype(float) - base_correct
        draws = patient_bootstrap_mean(d, pat, args.n_boot, cfg.seed)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append({"model": disp, "family": fam_of[disp],
                     "delta_accuracy": float(d.mean()),
                     "ci_low": float(lo), "ci_high": float(hi),
                     "excludes_zero": bool((lo > 0) == (hi > 0)),
                     "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())})
    #  The fast path is an identity, not an approximation. Check it against a
    #  naive case-level resample on one model, with the same RNG stream.
    d0 = (probs["CT-HMM, 3 latent states"].argmax(1) == y).astype(float) - base_correct
    codes, _ = pd.factorize(pat, sort=True)
    order = np.argsort(codes, kind="mergesort")
    blocks = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng_n = np.random.default_rng(cfg.seed)
    naive, sizes = np.empty(200), np.empty(200, int)
    for b in range(200):
        pick = rng_n.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        naive[b] = d0[idx].mean()
        sizes[b] = len(idx)
    fast = patient_bootstrap_mean(d0, pat, 200, cfg.seed)
    gap = float(np.abs(naive - fast).max())
    say(f"\n  fast bootstrap against a naive case-level resample, 200 draws:")
    say(f"    max absolute difference {gap:.2e}")
    say(f"    replicate patient count fixed at {len(blocks):,}; "
        f"replicate case total {sizes.min():,} to {sizes.max():,} "
        f"(point estimate {len(d0):,})")
    checks["the fast cluster bootstrap equals a naive case-level resample"] = gap < 1e-12
    checks["the bootstrap draws whole patients, not individual cases"] = bool(
        sizes.min() < len(d0) < sizes.max())

    con = pd.DataFrame(rows).sort_values("delta_accuracy", ascending=False)
    con.to_csv(out / "paired_contrasts.csv", index=False)
    say(frame(con.round(4)))
    n_excl = int(con[con.family != "baseline"].excludes_zero.sum())
    say(f"\n  non-baseline specifications beating the canonical rule with an")
    say(f"  interval excluding zero: {n_excl} of "
        f"{int((con.family != 'baseline').sum())}")

    # ------------------------------------------- where the margin comes from
    section("7. WHERE THE MARGIN OVER THE TRIVIAL RULE COMES FROM")
    say("  The contrasts above are pooled. Splitting them by the label recorded")
    say("  at the prediction point localises the whole effect.\n")
    cur = base["actual"].copy()
    #  The current label is the baseline's own prediction: persistence names it.
    cur_label = np.array(CLASSES)[Pb.argmax(1)]
    rows = []
    for disp in ["CT-HMM, 3 latent states", "Subscore GP, independent",
                 "Markov chain, discrete"]:
        pm = probs[disp].argmax(1)
        ok_m = (pm == y).astype(int)
        ok_b = base_correct.astype(int)
        same = (pm == Pb.argmax(1))
        for c in CLASSES:
            sel = cur_label == c
            rows.append({
                "model": disp, "current_label": c, "n": int(sel.sum()),
                "share_of_cases": float(sel.mean()),
                "model_accuracy": float(ok_m[sel].mean()),
                "baseline_accuracy": float(ok_b[sel].mean()),
                "cases_gained": int(ok_m[sel].sum() - ok_b[sel].sum()),
                "identical_predictions": float(same[sel].mean()),
            })
    dec = pd.DataFrame(rows)
    dec["share_of_total_gain"] = dec.groupby("model")["cases_gained"].transform(
        lambda s: s / s.sum() if s.sum() else np.nan)
    dec.to_csv(out / "margin_decomposition.csv", index=False)
    say(frame(dec.round(4)))

    hmm_dec = dec[dec.model == "CT-HMM, 3 latent states"].set_index("current_label")
    ind_share = float(hmm_dec.loc["Indeterminate", "share_of_total_gain"])
    say(f"\n  For the CT-HMM, {ind_share:.1%} of the total gain over the trivial rule")
    say(f"  comes from the {int(hmm_dec.loc['Indeterminate','n']):,} cases "
        f"({hmm_dec.loc['Indeterminate','share_of_cases']:.1%} of the data) where the")
    say("  currently recorded label is Indeterminate. On the other 90% the two")
    say(f"  make the same prediction "
        f"{hmm_dec.loc['TD','identical_predictions']:.1%} of the time when the current")
    say(f"  label is TD and {hmm_dec.loc['PIGD','identical_predictions']:.1%} when it is PIGD, "
        f"and on current-PIGD cases")
    say(f"  their accuracies differ by "
        f"{hmm_dec.loc['PIGD','model_accuracy'] - hmm_dec.loc['PIGD','baseline_accuracy']:+.4f}.")
    say("")
    say("  Read plainly: the fitted models do not predict progression better than")
    say("  the trivial rule. They decline to predict a class the trivial rule")
    say("  repeats, and that class is almost never the right answer.")
    checks["the margin decomposition accounts for every case"] = bool(
        dec.groupby("model")["n"].sum().eq(len(y)).all())

    # ----------------------------------------------------- Monte Carlo error
    section("8. MONTE CARLO ERROR")
    rep = {}
    for disp, (path, model, _, _) in {m["model"]: (m["source"], m["source_model"], None, None)
                                      for m in meta if "Monte Carlo" in m["derivation"]}.items():
        d = pd.read_csv(ROOT / path, low_memory=False)
        d = d[d.model == model].sort_values(["held_out_patient", "history_length"])
        P2 = derive_probabilities(d.reset_index(drop=True), cfg, args.n_draws // 2)
        g = pd.DataFrame({"patient": d["held_out_patient"].to_numpy(),
                          "n_history": d["history_length"].to_numpy()})
        g[PCOLS] = P2
        m2 = base[["patient", "n_history"]].merge(g, on=["patient", "n_history"], how="left")
        a_full = (probs[disp].argmax(1) == y).mean()
        a_half = (m2[PCOLS].to_numpy().argmax(1) == y).mean()
        rep[disp] = (a_full, a_half, abs(a_full - a_half))
    mc = pd.DataFrame([{"model": k, "accuracy_full_draws": v[0],
                        "accuracy_half_draws": v[1], "abs_difference": v[2]}
                       for k, v in rep.items()])
    mc.to_csv(out / "monte_carlo_error.csv", index=False)
    say(frame(mc.round(5)))
    worst = float(mc.abs_difference.max())
    say(f"\n  worst accuracy shift when halving the draws: {worst:.5f}")
    checks["halving the Monte Carlo draws moves accuracy by less than 0.005"] = worst < 0.005

    # ------------------------------------------------- against the committed
    section("9. AGAINST THE PREVIOUSLY COMMITTED PER-FAMILY FIGURES")
    say("  These will not match exactly, and the reasons are the two decisions")
    say("  in the module docstring: both-zero draws now go to Indeterminate,")
    say("  and the Monte Carlo is fresh. Shown so the difference is visible.\n")
    prev = {"CT-HMM, 3 latent states": 0.7539, "CT-HMM, 5 latent states": 0.7513,
            "Markov chain, discrete": 0.7474, "CTMC, interval-aware": 0.7453,
            CANONICAL: 0.7227, "Marginal null": 0.6293,
            "Subscore GP, independent": 0.7566, "Subscore GP, joint": 0.7566,
            "Joint subscore LMM": 0.7087, "Time-varying LMM, spline": 0.7093,
            "Time-varying LMM, constant": 0.7087}
    rows = []
    for k, v in prev.items():
        if k in acc.index:
            rows.append({"model": k, "committed": v, "harmonised": float(acc[k]),
                         "difference": float(acc[k]) - v})
    cmpf = pd.DataFrame(rows)
    cmpf.to_csv(out / "versus_committed.csv", index=False)
    say(frame(cmpf.round(4)))
    native = cmpf[~cmpf.model.str.contains("Subscore|LMM")]
    say(f"\n  natively-3-class models reproduce exactly: "
        f"{bool((native.difference.abs() < 1e-4).all())} "
        f"(max |diff| {native.difference.abs().max():.2e})")
    checks["models already on a 3-class vector reproduce their committed accuracy"] = bool(
        (native.difference.abs() < 1e-4).all())

    # --------------------------------------------------------------- figure
    section("10. FIGURE")
    #  Both panels share one ordering and one set of row labels, so the rows
    #  line up and the right panel needs no y-axis of its own.
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.06})
    t = tab.sort_values("accuracy").reset_index(drop=True)
    is_base = (t.family == "baseline").to_numpy()
    cols = np.where(is_base, MUTED, ACCENT)
    yy = np.arange(len(t))

    ax = axes[0]
    ax.barh(yy, t.accuracy, color=cols, height=0.64, alpha=0.92)
    ax.axvline(acc[CANONICAL], color=WARM, lw=1.4, ls="--", zorder=3)
    ax.text(acc[CANONICAL] - 0.004, len(t) - 0.4,
            f"persistence {acc[CANONICAL]:.3f}", color=WARM, fontsize=8.4,
            ha="right", va="top")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{m}" + ("  *" if b else "")
                        for m, b in zip(t.model, is_base)], fontsize=8.4)
    ax.set_xlim(0.55, 0.80)
    ax.set_xlabel("3-class accuracy, identical 3,361 held-out transitions", fontsize=9.4)
    ax.set_title("Every family under one decision rule", fontsize=10.6, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    ax = axes[1]
    ci = con.set_index("model")
    for i, m in enumerate(t.model):
        if m == CANONICAL:
            continue
        r = ci.loc[m]
        col = MUTED if r.family == "baseline" else ACCENT
        ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=2.2, solid_capstyle="butt")
        ax.plot(r.delta_accuracy, i, "o", ms=4.2, color=col)
    ax.axvline(0.0, color=WARM, lw=1.4, ls="--", zorder=3)
    ax.set_xlabel("accuracy difference from the persistence rule", fontsize=9.4)
    ax.set_title("Paired contrasts, patient-clustered 95% intervals",
                 fontsize=10.6, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.tick_params(axis="y", length=0)
    fig.text(0.012, 0.012, "*  baseline, not a fitted model",
             fontsize=8.2, color=INK_MUTED)

    p = _finish(fig, out / "fig1_harmonised_family_comparison.png")
    say(f"  {p.name}")
    tab.to_csv(out / "fig1_harmonised_family_comparison.csv", index=False)

    base.to_csv(out / "harmonised_predictions.csv", index=False)
    pd.DataFrame(meta).to_csv(out / "model_inventory.csv", index=False)

    # ------------------------------------------------------------- artefacts
    section("11. CHECKS")
    cf = pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()])
    cf.to_csv(out / "validation_checks.csv", index=False)
    say(frame(cf))
    n_ok = int(cf["passed"].sum())
    say(f"\n  {n_ok} of {len(cf)} checks passed")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (out / "summary.json").write_text(json.dumps({
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "node": platform.node(), "seed": cfg.seed,
            "n_monte_carlo_draws": args.n_draws, "n_bootstrap": args.n_boot,
            "elapsed_seconds": round(time.time() - started, 1),
            "git_commit": commit, "models_fitted": 0, "analysis_only": True,
        },
        "canonical_baseline": CANONICAL,
        "accuracy": {k: float(v) for k, v in acc.items()},
        "checks": {k: bool(v) for k, v in checks.items()},
        "code_sha256": {"scripts/79_harmonised_family_comparison.py": sha(__file__)},
    }, indent=1), encoding="utf-8")

    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "driver.log").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(q for q in out.rglob("*")
                   if q.is_file() and not q.is_symlink() and q.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(q)}  {q.relative_to(out)}\n" for q in files), encoding="utf-8")

    print(f"\nWrote {len(files)} files to {out}")
    return 0 if n_ok == len(cf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
