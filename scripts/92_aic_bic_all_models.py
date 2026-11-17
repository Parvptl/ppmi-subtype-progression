"""AIC and BIC for every model family in this project that has a maximised likelihood.

`scripts/65_model_selection_criteria.py` did this for three families: the
rich clinical profile, the observed-label chains and the CT-HMM. This script
re-derives those on the frozen panel and adds the families that came later or
were never put on the criteria: the **ratio-banded CT-HMM** at every band
width, the ratio-trajectory GPs, the bivariate subscore models (joint LMM,
time-varying spline LMMs, multi-output GPs) and the latent clinical states.

Every model is fitted ONCE on the full cohort, purely to obtain a likelihood at
its own maximum, which is what AIC and BIC need. No model is modified and no
held-out number is recomputed here.

What is comparable with what
----------------------------
AIC and BIC are only meaningful between models whose likelihood is the *same
object over the same data*. Six groups are internally valid, and nothing is
comparable across them:

  L1  joint PMF over whole subtype-label sequences (4,295 labels)
      CT-HMM against the observed-label chains given an initial-state
      distribution, which is what puts them on the CT-HMM's scale.
  L2  conditional PMF over transition pairs (3,361 pairs)
      CTMC against the discrete Markov chain.
  B   joint PMF over whole (subtype x ratio band) sequences, one group PER
      BAND WIDTH. Within a width the banded CT-HMM is compared with its own
      nested restriction (see below). Across widths the alphabet differs, so
      the likelihood is over a different object and the criteria do not
      transfer.
  P   density over the nine continuous clinical domains (38,486 values)
      the three nested rich-profile variants.
  R   density over the tremor/PIGD ratio (3,528 values, 821 patients)
      the four GP kernels.
  S   density over the two subscores jointly (8,590 values)
      joint LMM, the time-varying spline LMMs and the multi-output GPs --- all
      maximum likelihood (never REML) over exactly the same observations.
  C   density over the ten standardised clinical features (42,773 values)
      the latent-clinical-state CT-HMM at each state count. These are the
      rich profile's nine domains plus LEDD.

How the banded CT-HMM is made comparable
----------------------------------------
The banded model's likelihood is a mass over (subtype, band) pairs, the square
CT-HMM's a mass over subtypes alone, so their AICs are not on one scale. The
fix is the same device script 65 used for the chains. The banded emission
factorises as ``E[a, (s, b)] = M[a, s] * C[a, s, b]``; restricting
``C[a, s, b] = c[s, b]`` --- the band distribution does not depend on the
latent state --- gives a model nested inside the banded one whose joint
log-likelihood separates exactly:

    loglik_restricted = loglik_square_CT-HMM + sum_t log c[subtype_t, band_t]

because a band factor that is constant across latent states multiplies out of
the forward recursion. Its MLE is therefore the square CT-HMM's fit plus the
empirical within-subtype band frequencies. That restriction is the *null* the
band experiment was really testing --- "the band carries no information about
the latent state" --- and the two sit on one likelihood over one dataset, so
AIC and BIC between them are valid.

The restricted solution is also fed to the banded fit as an extra start, so
the larger model cannot lose to its own submodel through a bad optimum.

    python scripts/92_aic_bic_all_models.py --out <dir> [--family F ...]
    python scripts/92_aic_bic_all_models.py --out <dir> --combine

Families run independently and each writes its own CSV, so they can be run in
parallel into one --out directory and combined afterwards.
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402

#: The frozen panel every published result in this project was computed from.
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

#: Band widths: the pre-specified sweep of script 90 plus the 0.50 arm added by
#: the extended sweep, so the table covers every width ever fitted.
BAND_WIDTHS = (0.05, 0.10, 0.20, 0.30, 0.50)

#: Latent-clinical-state counts, as script 61 tested them.
STATE_COUNTS = (2, 3, 4, 5, 6, 8)

#: EM starts per latent-clinical-state fit, as script 61's development sweep
#: used. EM is badly multimodal on these features, so with too few starts the
#: log-likelihood is not even monotone in the state count --- a larger model
#: lands in a worse optimum than its own submodel found. Overridable with
#: --state-starts; the spread across starts is always reported.
STATE_STARTS = 8

FAMILIES = ("label", "latent_size", "band", "profile", "ratio_gp", "subscore",
            "states")


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------
def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def criteria(loglik: float, k: int, n: int) -> tuple[float, float]:
    """AIC and BIC. `n` is the number of terms in the likelihood."""
    return 2 * k - 2 * loglik, k * np.log(max(n, 2)) - 2 * loglik


def row(group, model, likelihood, loglik, k, n, **extra) -> dict:
    aic, bic = criteria(loglik, k, n)
    out = {"group": group, "model": model, "likelihood": likelihood,
           "loglik": float(loglik), "k": int(k), "n_terms": int(n),
           "aic": float(aic), "bic": float(bic)}
    out.update(extra)
    return out


class Log:
    """Collects everything printed so the family's report can be written out."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def section(self, text: str) -> None:
        self("\n" + "=" * 78)
        self(text)
        self("=" * 78)

    def frame(self, frame: pd.DataFrame, **kw) -> None:
        self("    " + frame.to_string(**kw).replace("\n", "\n    "))


def load_panel(cfg, say: Log) -> pd.DataFrame:
    path = cfg.path("outputs") / "visit_panel.csv"
    got = md5(path)
    if got != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {got} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel {path.name}  md5 {got}  [MATCH]")
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
#  L: the categorical subtype label
# ---------------------------------------------------------------------------
def fit_square_cthmm(cfg, data, say: Log):
    """The published next-visit CT-HMM, refitted, with the winning theta kept.

    `hmm_stability.fit_multistart` over `next_visit.audited_starts` is the same
    objective, the same starts and the same optimiser settings as
    `hmm.fit_hmm`; it additionally returns every start's parameter vector,
    which the banded family needs for its nested warm start.
    """
    from ppmi_subtype import hmm_stability, loo, next_visit as nv

    starts = nv.audited_starts(len(data.states), len(data.states),
                               cfg.model["emission_floor"], cfg.seed,
                               cfg.model["n_restarts"])
    t = time.time()
    fit, records = hmm_stability.fit_multistart(
        data.sequences, data.states, n_hidden=len(data.states),
        emission_floor=cfg.model["emission_floor"], starts=starts,
        state_names=loo.STATE_NAMES)
    best = max(records, key=lambda r: r.loglik)
    say(f"  CT-HMM fitted in {time.time() - t:.1f}s  "
        f"(best of {len(records)} starts, spread "
        f"{max(r.loglik for r in records) - min(r.loglik for r in records):.3f})")
    return fit, np.asarray(best.theta, dtype=float)


def family_label(cfg, panel, say: Log) -> list[dict]:
    from ppmi_subtype import ctmc, loo

    say.section("L. MODELS OF THE CATEGORICAL SUBTYPE LABEL")
    states = [cfg.subtype.labels[s] for s in loo.STATE_NAMES]
    data = loo.prepare(panel, states)
    say(f"  cohort {len(data.sequences):,} patients, "
        f"{data.n_transitions:,} transitions")

    fit, _ = fit_square_cthmm(cfg, data, say)
    n_obs = fit.n_observations
    rows = [row("L1 joint PMF over label sequences", "CT-HMM (3 latent states)",
                "joint PMF over label sequences", fit.loglik, fit.n_params, n_obs,
                converged=bool(fit.converged),
                notes="6 rates + 6 emission + 2 initial")]

    # -- the latent layer's two other specifications ------------------------
    # The published model constrains P(label a | state a) >= 0.5. Dropping the
    # floor, and allowing more latent states than labels, are both the same
    # likelihood over the same 4,295 labels, so they belong in this group. The
    # floored fit is a CONSTRAINED maximum, so its likelihood is a lower bound
    # on what the unconstrained three-state family can reach.
    from ppmi_subtype import hmm

    for n_hidden, floor, label, note in (
            (3, None, "CT-HMM (3 latent states, no emission floor)",
             "unconstrained ML: degenerate (see hmm.py) but a valid maximum"),
            (5, None, "CT-HMM (5 latent states, no emission floor)",
             "the 5-hidden sensitivity model of script 10")):
        t = time.time()
        extra = hmm.fit_hmm(data.sequences, states, state_names=loo.STATE_NAMES,
                            seed=cfg.seed, n_restarts=cfg.model["n_restarts"],
                            emission_floor=floor, n_hidden=n_hidden)
        say(f"  {label}: loglik {extra.loglik:,.2f}  k {extra.n_params}  "
            f"({time.time() - t:.0f}s)")
        rows.append(row("L1 joint PMF over label sequences", label,
                        "joint PMF over label sequences", extra.loglik,
                        extra.n_params, extra.n_observations,
                        converged=bool(extra.converged), notes=note))

    # -- the observed-label chains, on their own conditional likelihood -----
    chain = ctmc.fit_ctmc(data.pairs, states, state_names=loo.STATE_NAMES,
                          seed=cfg.seed, n_restarts=cfg.model["n_restarts"])
    rows.append(row("L2 conditional PMF over pairs", "CTMC (interval-aware)",
                    "conditional PMF over pairs", chain.loglik, chain.n_params,
                    chain.n_pairs, converged=True,
                    notes="likelihood conditional on the previous observed state"))

    dtmc = ctmc.fit_dtmc(data.pairs, states)
    index = {s: k for k, s in enumerate(states)}
    i = data.pairs["from_state"].map(index).to_numpy(int)
    j = data.pairs["to_state"].map(index).to_numpy(int)
    ll_discrete = float(np.log(np.maximum(dtmc[i, j], 1e-300)).sum())
    k_discrete = len(states) * (len(states) - 1)
    rows.append(row("L2 conditional PMF over pairs", "Markov (discrete)",
                    "conditional PMF over pairs", ll_discrete, k_discrete,
                    len(data.pairs), converged=True,
                    notes="evaluated at the Laplace-smoothed estimate, so very "
                          "slightly understated"))

    # -- the chains given an initial state, which is the CT-HMM's scale -----
    first = np.array([s["labels"][0] for s in data.sequences])
    pi_hat = np.bincount(first, minlength=len(states)) / len(first)
    ll_init = float(np.log(np.maximum(pi_hat[first], 1e-300)).sum())
    say(f"  initial-state distribution {np.round(pi_hat, 4).tolist()}  "
        f"loglik {ll_init:,.2f}  (+2 parameters)")
    if n_obs - len(data.pairs) != len(data.sequences):
        raise SystemExit("the CT-HMM's extra likelihood terms are not the "
                         "patients' first visits")

    for name, ll_pairs, k_pairs in (
            ("CTMC (interval-aware) + initial state", chain.loglik, chain.n_params),
            ("Markov (discrete) + initial state", ll_discrete, k_discrete)):
        rows.append(row("L1 joint PMF over label sequences", name,
                        "joint PMF over label sequences", ll_pairs + ll_init,
                        k_pairs + 2, n_obs, converged=True,
                        notes="chain plus the empirical first-visit distribution"))
    return rows


# ---------------------------------------------------------------------------
#  B: the ratio-banded CT-HMM, one group per band width
# ---------------------------------------------------------------------------
def band_restriction(square_loglik: float, square_theta: np.ndarray,
                     alphabet, codes: np.ndarray, n_hidden: int):
    """The state-independent-band restriction: its loglik, k, and a warm start.

    ``C[a, s, b] = c[s, b]`` factors out of the forward recursion, so the joint
    maximum is the square model's fit plus the empirical within-subtype band
    frequencies. Returns ``(loglik, k, k_identified, theta, n_empty)``.
    """
    from ppmi_subtype import hmm_banded

    counts = np.bincount(codes, minlength=alphabet.n_categories).astype(float)
    ll_bands = 0.0
    k_free = 0
    n_empty = 0
    band_logits: list[float] = []
    for block in alphabet.blocks:
        cols = np.asarray(list(block), dtype=int)
        share = counts[cols] / counts[cols].sum()
        seen = counts[cols] > 0
        ll_bands += float((counts[cols][seen] * np.log(share[seen])).sum())
        k_free += len(cols) - 1
        n_empty += int((~seen).sum())
        with np.errstate(divide="ignore"):
            logits = np.log(share) - np.log(share[0])
        band_logits.extend(np.clip(logits[1:], -hmm_banded.BAND_LOGIT_BOUND,
                                   hmm_banded.BAND_LOGIT_BOUND).tolist())

    # The square layout is rates | emission | initial; the banded layout is
    # rates | M | bands | initial, with the same pinned-logit convention.
    n_rates = n_hidden * (n_hidden - 1)
    n_emit = n_hidden * (len(alphabet.blocks) - 1)
    theta = np.concatenate([square_theta[:n_rates + n_emit],
                            np.tile(np.asarray(band_logits, dtype=float), n_hidden),
                            square_theta[n_rates + n_emit:]])
    if len(theta) != hmm_banded.n_params(alphabet, n_hidden):
        raise SystemExit("the warm start does not match the banded parameter layout")
    return (square_loglik + ll_bands, 14 + k_free,
            14 + k_free - n_empty, theta, n_empty)


def family_band(cfg, panel, say: Log) -> list[dict]:
    from ppmi_subtype import hmm_banded, loo, next_visit_banded as nvb, ratio_bands as rb

    say.section("B. THE RATIO-BANDED CT-HMM, ONE GROUP PER BAND WIDTH")
    say("  Likelihood is a joint PMF over whole (subtype x ratio band) sequences.")
    say("  Within a width the banded model is compared with its own nested")
    say("  restriction, in which the band distribution does not depend on the")
    say("  latent state. Across widths the alphabet differs and the criteria")
    say("  do not transfer.")

    states = [cfg.subtype.labels[s] for s in loo.STATE_NAMES]
    data = loo.prepare(panel, states)
    square, square_theta = fit_square_cthmm(cfg, data, say)
    bdata = nvb.prepare(panel, data, cfg.subtype)
    n_hidden = len(data.states)

    rows = []
    for width in BAND_WIDTHS:
        scheme, alphabet, _ = nvb.fold_alphabet(
            bdata, None, rb.MIN_TAIL_OBSERVATIONS, rb.MIN_TAIL_PATIENTS, width)
        sequences = nvb.joint_sequences(bdata, alphabet, bdata.patients)
        codes = np.concatenate([s["labels"] for s in sequences])
        group = f"B width {width:.2f} (joint PMF over subtype x band)"

        ll_r, k_r, k_r_id, warm, n_empty = band_restriction(
            square.loglik, square_theta, alphabet, codes, n_hidden)

        starts = np.vstack([
            hmm_banded.starts_for(alphabet, n_hidden, cfg.model["emission_floor"],
                                  cfg.seed, cfg.model["n_restarts"]),
            warm[None, :]])
        t = time.time()
        fit, records = hmm_banded.fit_banded(
            sequences, alphabet, bdata.subtype_names, n_hidden=n_hidden,
            emission_floor=cfg.model["emission_floor"], starts=starts)
        warm_won = int(max(range(len(records)),
                           key=lambda s: records[s].loglik)) == len(records) - 1

        if fit.n_observations != square.n_observations:
            raise SystemExit("the banded and square likelihoods cover different visits")
        if fit.loglik < ll_r - 1e-6:
            raise SystemExit(
                f"width {width}: the banded fit ({fit.loglik:.4f}) is below its own "
                f"nested restriction ({ll_r:.4f}); the optimum was not found")

        say(f"\n  width {width:.2f}  {alphabet.n_categories} categories "
            f"(blocks {[len(b) for b in alphabet.blocks]}), "
            f"{len(scheme.tail_lows)} tail bands, fitted in {time.time() - t:.1f}s")
        say(f"    bands depend on the latent state   loglik {fit.loglik:>12,.2f}  "
            f"k {fit.n_params:>4}  (warm start won: {warm_won})")
        say(f"    bands do not (nested restriction)  loglik {ll_r:>12,.2f}  "
            f"k {k_r:>4}")
        say(f"    gain {fit.loglik - ll_r:,.2f} loglik points for "
            f"{fit.n_params - k_r} extra parameters; "
            f"{n_empty} category/ies empty in the full cohort")

        rows.append(row(group, f"Banded CT-HMM, state-dependent bands "
                               f"(width {width:.2f})",
                        "joint PMF over subtype x band sequences",
                        fit.loglik, fit.n_params, fit.n_observations,
                        k_identified=int(fit.n_params - n_hidden * n_empty),
                        converged=bool(fit.converged),
                        n_categories=int(alphabet.n_categories),
                        notes=f"warm_start_won={warm_won}"))
        rows.append(row(group, f"Banded CT-HMM, state-independent bands "
                               f"(width {width:.2f})",
                        "joint PMF over subtype x band sequences",
                        ll_r, k_r, fit.n_observations,
                        k_identified=int(k_r_id), converged=bool(square.converged),
                        n_categories=int(alphabet.n_categories),
                        notes="square CT-HMM + empirical within-subtype band "
                              "frequencies; nested in the row above"))
    return rows


# ---------------------------------------------------------------------------
#  P: the rich clinical profile
# ---------------------------------------------------------------------------
def family_profile(cfg, panel, say: Log) -> list[dict]:
    from ppmi_subtype import clinical_profile as cp, clinical_states as cs

    say.section("P. THE RICH CLINICAL PROFILE (NINE CONTINUOUS DOMAINS)")
    say("  Likelihood is a probability DENSITY on the standardised scale. The")
    say("  three variants are nested and share one likelihood object.")

    sequences = cp.build_sequences(cs.build_clinical_panel(cfg, panel))
    z = cp.Standardiser.fit(sequences).apply(sequences)
    n_terms = int(np.isfinite(np.vstack([s["Y"] for s in z])).sum())
    say(f"  cohort {len(sequences):,} patients, {n_terms:,} observed values")

    rows = []
    for name, kwargs in ((cp.MODEL_FULL, {"use_rate": True, "use_medication": True}),
                         (cp.MODEL_NO_RATE, {"use_rate": False, "use_medication": True}),
                         (cp.MODEL_NO_MED, {"use_rate": True, "use_medication": False})):
        t = time.time()
        fit = cp.fit(z, **kwargs)
        say(f"\n    {name}")
        say(f"      loglik {fit.loglik:,.2f}  k {fit.n_params}  "
            f"({time.time() - t:.0f}s, converged={fit.converged})")
        rows.append(row("P density over 9 clinical domains", name,
                        "density over 9 continuous domains", fit.loglik,
                        fit.n_params, n_terms, converged=bool(fit.converged),
                        notes="density on the standardised scale"))
    return rows


# ---------------------------------------------------------------------------
#  R: the ratio-trajectory GPs
# ---------------------------------------------------------------------------
def family_ratio_gp(cfg, panel, say: Log) -> list[dict]:
    from ppmi_subtype import gpr, gpr_loo, trajectory as tj

    say.section("R. GAUSSIAN-PROCESS MODELS OF THE TREMOR/PIGD RATIO")
    say("  Likelihood is the marginal density of every finite ratio. The two")
    say("  mean-function coefficients are profiled out by the fitter, so they")
    say("  are counted as parameters here. K0 = const + linear (the maximum-")
    say("  likelihood counterpart of the hierarchical growth model), K1 =")
    say("  const + RBF, K2 = K0 + RBF, K3 = K0 + Matern-3/2.")

    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    sequences = gpr_loo.prepare(tj.build_cohort(traj, cfg, criterion=None)).sequences()
    n_terms = int(sum(len(s["y"]) for s in sequences))
    say(f"  cohort {len(sequences):,} patients, {n_terms:,} finite ratios")

    rows = []
    for kernel in cfg.gpr["kernels"]:
        t = time.time()
        fit = gpr.fit_gpr(sequences, kernel, n_restarts=cfg.gpr["n_restarts"],
                          seed=cfg.seed, jitter=cfg.gpr["jitter"],
                          max_iter=cfg.gpr["max_iter"])
        k = gpr.n_parameters(kernel) + 2
        say(f"    {kernel}  loglik {fit.log_marginal_likelihood:>12,.2f}  k {k:>3}  "
            f"({time.time() - t:.0f}s)")
        rows.append(row("R density over the ratio", f"GP {kernel}",
                        "marginal density over the ratio",
                        fit.log_marginal_likelihood, k, n_terms,
                        converged=bool(getattr(fit, "converged", True)),
                        notes=f"{gpr.n_parameters(kernel)} kernel + 2 mean "
                              f"(profiled) parameters"))
    return rows


# ---------------------------------------------------------------------------
#  S: the bivariate subscore models
# ---------------------------------------------------------------------------
def family_subscore(cfg, panel, say: Log) -> list[dict]:
    from ppmi_subtype import (jointlmm as J, mogp as M, timevarying as TV,
                              timevarying_ext as TVX, trajectory as tj)

    say.section("S. MODELS OF THE TWO SUBSCORES JOINTLY")
    say("  Likelihood is a density over the tremor and PIGD subscores of every")
    say("  visit. Every fit here is MAXIMUM LIKELIHOOD, never REML: a REML")
    say("  log-likelihood cannot be compared across mean structures, and the")
    say("  spline models differ in exactly that. The multi-output GPs profile")
    say("  their four mean coefficients out, so those are counted here.")

    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    frame = traj.merge(
        panel[["PATNO", "EVENT_ID", *J.OUTCOMES]].drop_duplicates(
            ["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    if frame[list(J.OUTCOMES)].isna().any().any():
        raise SystemExit("a visit is missing a subscore after the merge")
    data = J.pack(frame)
    n_terms = int(data.n_visits * 2)
    say(f"  cohort {len(data.patients):,} patients, {data.n_visits:,} visits, "
        f"{n_terms:,} observed values")
    group = "S density over both subscores"

    rows = []

    # -- the joint LMM, and the spline mean structures around it ------------
    lmm = J.fit_joint_lmm(data, reml=False)
    k_lmm = 4 + 3 + 3
    say(f"\n    joint LMM (random intercepts)        loglik "
        f"{-0.5 * lmm.neg2ll:>12,.2f}  k {k_lmm:>3}")
    rows.append(row(group, "Joint LMM, random intercepts",
                    "density over both subscores", -0.5 * lmm.neg2ll, k_lmm,
                    n_terms, converged=bool(lmm.converged),
                    notes="ML fit; linear mean, correlated residuals"))

    for name, knots in TV.KNOT_SETS.items():
        for random_slopes in (False, True):
            t = time.time()
            fit = TVX.fit_basis(data, TVX.linear_spline(knots),
                                random_slopes=random_slopes, reml=False)
            label = (f"Spline LMM, knots {name}"
                     f"{' + random slopes' if random_slopes else ''}")
            say(f"    {label:<48} loglik {-0.5 * fit.neg2ll:>12,.2f}  "
                f"k {fit.n_params:>3}  ({time.time() - t:.0f}s)")
            rows.append(row(group, label, "density over both subscores",
                            -0.5 * fit.neg2ll, fit.n_params, n_terms,
                            converged=bool(fit.converged),
                            notes=f"ML fit; knots {knots}"))
            # No knots and no random slopes IS the joint LMM, by a different
            # code path. If the two disagree, one of them is wrong.
            if name == "constant" and not random_slopes:
                gap = abs(fit.neg2ll - lmm.neg2ll)
                say(f"      check: reproduces the joint LMM to {gap:.2e} in -2logL")
                if gap > 1e-3:
                    raise SystemExit(
                        "the knot-free spline LMM does not reproduce the joint "
                        f"LMM ({fit.neg2ll} vs {lmm.neg2ll})")

    # -- the multi-output GPs ----------------------------------------------
    # Packed from the same per-patient arrays the LMMs were fitted on, so the
    # two routes cannot disagree about which observations are in the likelihood.
    packed = M.pack([{"t": data.times[p], "y": data.responses[p]}
                     for p in data.patients])
    if int(packed.n_obs) != n_terms:
        raise SystemExit(f"the multi-output GPs cover {packed.n_obs} values, "
                         f"the LMMs {n_terms}")
    for kernel in ("K0", "K1", "K2", "K3"):
        for coregion in ("full", "diag"):
            t = time.time()
            fit = M.fit_mogp(packed, kernel, coregion, n_restarts=cfg.gpr["n_restarts"],
                             max_iter=cfg.gpr["max_iter"], jitter=cfg.gpr["jitter"],
                             seed=cfg.seed)
            k = len(M.parameter_layout(kernel, coregion)) + 4
            label = f"Multi-output GP {kernel}/{coregion}"
            say(f"    {label:<48} loglik {-fit.nll:>12,.2f}  k {k:>3}  "
                f"({time.time() - t:.0f}s)")
            rows.append(row(group, label, "density over both subscores",
                            -fit.nll, k, n_terms, converged=bool(fit.converged),
                            notes=f"{k - 4} kernel + 4 mean (profiled) parameters"))
    return rows


# ---------------------------------------------------------------------------
#  C: the latent clinical states
# ---------------------------------------------------------------------------
def family_states(cfg, panel, say: Log, n_starts: int = STATE_STARTS) -> list[dict]:
    from ppmi_subtype import clinical_states as cs

    say.section("C. THE LATENT-CLINICAL-STATE CT-HMM, BY STATE COUNT")
    say("  Likelihood is a Gaussian density over the standardised clinical")
    say(f"  features, missing entries marginalised. {n_starts} EM starts per")
    say("  fit; the spread across starts is reported because EM is not stably")
    say("  identified here and the criteria inherit that uncertainty.")

    sequences = cs.build_sequences(cs.build_clinical_panel(cfg, panel))
    z = cs.Standardiser.fit(sequences).apply(sequences)
    n_terms = int(np.isfinite(np.vstack([s["X"] for s in z])).sum())
    say(f"  cohort {len(sequences):,} patients, {n_terms:,} observed values")

    rows = []
    for n_states in STATE_COUNTS:
        t = time.time()
        fit, records = cs.fit_em_multistart(z, n_states, seed=cfg.seed,
                                            n_starts=n_starts)
        spread = (max(r["loglik"] for r in records)
                  - min(r["loglik"] for r in records))
        say(f"    {n_states} states  loglik {fit.loglik:>12,.2f}  "
            f"k {fit.n_params:>4}  start spread {spread:,.1f}  "
            f"({time.time() - t:.0f}s)")
        rows.append(row("C density over 10 clinical features",
                        f"Latent clinical states, {n_states} states",
                        "density over 10 standardised features", fit.loglik,
                        fit.n_params, n_terms, converged=bool(fit.converged),
                        notes=f"{n_starts} EM starts, loglik spread "
                              f"{spread:.1f}"))
    return rows


# ---------------------------------------------------------------------------
#  L1 and L3: smaller latent chains, and a two-label alphabet
# ---------------------------------------------------------------------------
def fit_audited(cfg, sequences, states, n_hidden, floor, state_names, say: Log,
                label: str):
    """One audited multi-start fit, from the project's starts plus dispersed ones.

    `hmm.fit_hmm` refuses `n_hidden < n_observed`, so the two-latent-state chain
    over three labels cannot go through it. `hmm_stability.fit_multistart` runs
    the identical optimisation --- same objective, same bounds, same options ---
    and takes the starts explicitly, so the fit is the one `fit_hmm` would do if
    its guard allowed the configuration. Unconstrained fits (no emission floor)
    are multimodal, so the project's four audited starts are supplemented with
    eight dispersed ones and the spread is reported.
    """
    from ppmi_subtype import hmm_stability, next_visit as nv

    n_observed = len(states)
    starts = np.vstack([
        nv.audited_starts(n_hidden, n_observed, floor, cfg.seed,
                          cfg.model["n_restarts"]),
        hmm_stability.dispersed_starts(n_hidden, n_observed, floor, 8, cfg.seed)])
    t = time.time()
    fit, records = hmm_stability.fit_multistart(
        sequences, states, n_hidden=n_hidden, emission_floor=floor,
        starts=starts, state_names=state_names)
    spread = max(r.loglik for r in records) - min(r.loglik for r in records)
    say(f"    {label:<46} loglik {fit.loglik:>10,.2f}  k {fit.n_params:>3}  "
        f"n {fit.n_observations:>6,}  (spread {spread:,.2f}, {time.time() - t:.0f}s)")
    return fit, spread


def family_latent_size(cfg, panel, say: Log) -> list[dict]:
    from ppmi_subtype import loo

    say.section("L1 / L3. SMALLER LATENT CHAINS, AND A TWO-LABEL ALPHABET")
    labels = cfg.subtype.labels
    rows = []

    # -- 2 latent states over the three recorded labels: still group L1 -----
    say("\n  Two latent states, three recorded labels. Same 4,295 observations")
    say("  as every other model in group L1, so the criteria are comparable")
    say("  with the 3- and 5-state chains. The emission floor is undefined")
    say("  when the counts differ, so this fit is unconstrained, and its two")
    say("  latent states are interchangeable (label switching does not move")
    say("  the likelihood).")
    states3 = [labels[s] for s in loo.STATE_NAMES]
    data3 = loo.prepare(panel, states3)
    fit, spread = fit_audited(cfg, data3.sequences, states3, 2, None,
                              loo.STATE_NAMES, say,
                              "2 latent states x 3 labels")
    rows.append(row("L1 joint PMF over label sequences",
                    "CT-HMM (2 latent states, 3 labels, no emission floor)",
                    "joint PMF over label sequences", fit.loglik, fit.n_params,
                    fit.n_observations, converged=bool(fit.converged),
                    notes=f"2 rates + 4 emission + 1 initial; start spread "
                          f"{spread:.2f}"))

    # -- the two-label alphabet: a DIFFERENT dataset, hence its own group ---
    say("\n  Two recorded labels. Indeterminate visits are removed, which")
    say("  changes the dataset: fewer observations, and the gaps now run")
    say("  between consecutive TD/PIGD visits, skipping over any Indeterminate")
    say("  visit in between. These rows are therefore NOT comparable with")
    say("  group L1 --- only with each other.")
    panel2 = panel.copy()
    panel2.loc[panel2["subtype"] == labels["Indeterminate"], "subtype"] = np.nan
    states2 = [labels["TD"], labels["PIGD"]]
    names2 = ["TD", "PIGD"]
    data2 = loo.prepare(panel2, states2)
    dropped = int((panel["subtype"] == labels["Indeterminate"]).sum())
    say(f"\n    {dropped:,} Indeterminate visits removed; "
        f"{len(data2.sequences):,} patients keep two or more labelled visits, "
        f"{data2.n_transitions:,} transitions")

    for n_hidden, floor, label, note in (
            (2, cfg.model["emission_floor"],
             "CT-HMM (2 latent states, 2 labels, floor 0.5)",
             "square, floored exactly as the published 3-state model is"),
            (2, None, "CT-HMM (2 latent states, 2 labels, no emission floor)",
             "unconstrained ML over the same two-label data"),
            (3, None, "CT-HMM (3 latent states, 2 labels, no emission floor)",
             "comparator: more latent states than labels, same data")):
        fit, spread = fit_audited(cfg, data2.sequences, states2, n_hidden, floor,
                                  names2, say, label.split("(")[1].rstrip(")"))
        rows.append(row("L3 joint PMF over TD/PIGD-only label sequences", label,
                        "joint PMF over 2-label sequences", fit.loglik,
                        fit.n_params, fit.n_observations,
                        converged=bool(fit.converged),
                        notes=f"{note}; start spread {spread:.2f}"))
    return rows


RUNNERS = {"label": family_label, "band": family_band, "profile": family_profile,
           "ratio_gp": family_ratio_gp, "subscore": family_subscore,
           "states": family_states, "latent_size": family_latent_size}


# ---------------------------------------------------------------------------
#  Combine
# ---------------------------------------------------------------------------
COLUMNS = ["family", "group", "model", "likelihood", "loglik", "k", "k_identified",
           "n_terms", "aic", "bic", "delta_aic_in_group", "delta_bic_in_group",
           "converged", "n_categories", "notes"]


def combine(out: Path, say: Log) -> pd.DataFrame:
    parts = sorted(out.glob("family_*.csv"))
    if not parts:
        raise SystemExit(f"no family_*.csv in {out}; run the families first")
    table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    for _, sub in table.groupby("group", sort=False):
        table.loc[sub.index, "delta_aic_in_group"] = sub["aic"] - sub["aic"].min()
        table.loc[sub.index, "delta_bic_in_group"] = sub["bic"] - sub["bic"].min()
    for column in COLUMNS:
        if column not in table:
            table[column] = np.nan
    table = table[COLUMNS].sort_values(["family", "group", "aic"])
    table.to_csv(out / "aic_bic_all_models.csv", index=False)

    say.section("ALL MODELS, RANKED WITHIN EACH COMPARABLE GROUP")
    say("  Lower is better. Delta is to the best model IN THE SAME GROUP; it is")
    say("  meaningless between groups, which sit on different likelihoods.")
    for group, sub in table.groupby("group", sort=False):
        say(f"\n  {group}   (n = {int(sub['n_terms'].iloc[0]):,} terms)")
        say.frame(sub[["model", "loglik", "k", "aic", "bic",
                       "delta_aic_in_group", "delta_bic_in_group"]].round(2),
                  index=False)
    say(f"\n  {len(parts)} families combined: "
        f"{', '.join(p.stem.replace('family_', '') for p in parts)}")
    return table


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", action="append", choices=FAMILIES,
                    help="run only this family; repeatable. Default: all of them.")
    ap.add_argument("--state-starts", type=int, default=STATE_STARTS,
                    help="EM starts per latent-clinical-state fit")
    ap.add_argument("--combine", action="store_true",
                    help="combine the family_*.csv files already in --out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    say = Log()
    started = time.time()
    cfg = load_config()

    if args.combine:
        table = combine(out, say)
        provenance(out, cfg, say, started, table)
        (out / "report.txt").write_text("\n".join(say.lines), encoding="utf-8")
        manifest(out)
        print(f"\nWrote {out}")
        return 0

    say.section("INPUT")
    panel = load_panel(cfg, say)
    for family in (args.family or list(FAMILIES)):
        rows = (family_states(cfg, panel, say, args.state_starts)
                if family == "states" else RUNNERS[family](cfg, panel, say))
        frame = pd.DataFrame(rows)
        frame.insert(0, "family", family)
        frame.to_csv(out / f"family_{family}.csv", index=False)
        (out / f"report_{family}.txt").write_text("\n".join(say.lines),
                                                  encoding="utf-8")
        print(f"\nWrote {out / f'family_{family}.csv'}")
    return 0


def provenance(out: Path, cfg, say: Log, started: float, table: pd.DataFrame) -> None:
    say.section("PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(), "seed": cfg.seed,
           "panel_md5": EXPECTED_PANEL_MD5,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                              # noqa: BLE001
        env["git_commit"] = "unknown"
    for key, value in env.items():
        say(f"  {key:<18} {value}")

    code = [Path(__file__)] + [ROOT / "src/ppmi_subtype" / f for f in (
        "hmm.py", "hmm_banded.py", "ctmc.py", "clinical_profile.py",
        "clinical_states.py", "gpr.py", "mogp.py", "jointlmm.py",
        "timevarying.py", "timevarying_ext.py", "ratio_bands.py")]
    (out / "summary.json").write_text(json.dumps(
        {"environment": env,
         "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
         "criteria": table.to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")


def manifest(out: Path) -> None:
    skip = {"MANIFEST.sha256"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),
        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
