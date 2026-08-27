"""Stage 4 - hierarchical Bayesian growth model, and comparison with stage 3.

Implements the model specified in the proposal (pages 13-15):

    R_it^obs = R_it + eps^obs
    R_it     = (B0 + b0i) + (B1 + b1i) t + eps^proc
    b0i ~ N(0, tau0^2),  b1i ~ N(0, tau1^2)
    P(TD) = P(R > 1.15),  P(Ind) = P(0.90 <= R <= 1.15),  P(PIGD) = P(R < 0.90)

and compares it against the hidden Markov model from stage 3 on exactly the
same held-out transitions. Stage 3 is not modified.

    python scripts/04_hierarchical_bayes.py [--quick]

Writes to outputs/hbayes/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import hbayes, hmm, splitting  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

STATES = [1.0, 2.0, 3.0]
NAMES = ["TD", "Indeterminate", "PIGD"]
CODE_TO_NAME = dict(zip(STATES, NAMES))

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def frame(df: pd.DataFrame, index: bool = False) -> str:
    return "    " + df.to_string(index=index).replace("\n", "\n    ")


def transition_times(panel: pd.DataFrame, ratio_only: bool) -> list[tuple]:
    """Keys identifying each scoreable transition as (PATNO, from_time, to_time)."""
    df = panel[panel["subtype"].notna()]
    if ratio_only:
        df = df[df["ratio"].notna() & np.isfinite(df["ratio"])]
    df = df.sort_values(["PATNO", "years_from_baseline"])
    keys = []
    for pid, g in df.groupby("PATNO", sort=False):
        t = g["years_from_baseline"].to_numpy()
        for k in range(1, len(t)):
            keys.append((pid, round(float(t[k - 1]), 3), round(float(t[k]), 3)))
    return keys


def score(probs: np.ndarray, truth: list[str], label: str) -> dict:
    probs = np.clip(probs, 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    j = np.array([NAMES.index(x) for x in truth])
    p_true = probs[np.arange(len(j)), j]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(j)), j] = 1.0
    return {
        "model": label,
        "n": len(j),
        "loglik_per_pair": float(np.log(p_true).mean()),
        "accuracy": float((probs.argmax(axis=1) == j).mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }


def head_to_head(panel, train_panel, test_panel, cfg, seed, max_draws):
    """Fit both models on training patients, score both on shared transitions."""
    post = hbayes.fit_growth_model(
        hbayes.build_growth_data(train_panel), n_draws=1500, burn_in=500, seed=seed
    )
    _, hb_preds = hbayes.evaluate(post, test_panel, max_draws=max_draws, seed=seed)
    hb_keys = transition_times(test_panel, ratio_only=True)
    hb_preds = hb_preds.assign(_key=hb_keys)

    fit = hmm.fit_hmm(
        hmm.build_sequences(train_panel, STATES), STATES, NAMES, seed=seed,
        n_restarts=2, emission_floor=cfg.model["emission_floor"],
    )
    hp = hmm.predictive_distributions(fit, hmm.build_sequences(test_panel, STATES))
    hp = hp.assign(_key=transition_times(test_panel, ratio_only=False))

    common = set(hb_keys) & set(hp["_key"])
    hb_c = hb_preds[hb_preds["_key"].isin(common)]
    hp_c = hp[hp["_key"].isin(common)]

    cols = [f"p_{n}" for n in NAMES]
    return (
        post,
        score(hp_c[cols].to_numpy(), [CODE_TO_NAME[x] for x in hp_c["to_state"]],
              "HMM (latent states)"),
        score(hb_c[cols].to_numpy(), hb_c["observed"].tolist(),
              "Hierarchical Bayesian growth"),
        len(common),
    )


def main() -> int:
    quick = "--quick" in sys.argv
    cfg = load_config()
    out_dir = cfg.path("outputs") / "hbayes"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        print(f"{panel_path} not found; run scripts/01_build_panel.py first", file=sys.stderr)
        return 1
    panel = pd.read_csv(panel_path)

    # ------------------------------------------------------------ data
    section("1. DATA FOR THE GROWTH MODEL")
    labelled = panel[panel["subtype"].notna()]
    defined = labelled[labelled["ratio"].notna() & np.isfinite(labelled["ratio"])]
    say(f"  labelled visits                      {len(labelled):,}")
    say(f"  with a finite TD/PIGD ratio          {len(defined):,}  ({len(defined)/len(labelled):.1%})")
    say(f"  without (PIGD subscore is zero)      {len(labelled)-len(defined):,}")
    dropped = labelled[~labelled.index.isin(defined.index)]["subtype_name"].value_counts()
    say(f"  subtype of the dropped visits        {dropped.to_dict()}")
    say("\n  The growth model is defined on the ratio, so those visits cannot enter it.")
    say("  They are overwhelmingly TD, so the loss is not random. The Markov models")
    say("  in stage 3 do not have this restriction, which is why the comparison below")
    say("  is restricted to transitions both models can score.")

    say("\n  Ratio distribution (finite values):")
    say(frame(defined["ratio"].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_frame().T))

    # ------------------------------------------------------------- fit
    section("2. FITTED MODEL (training patients)")
    manifest = splitting.split_patients(panel, cfg.splits["train"], cfg.seed)
    train_ids = set(manifest.loc[manifest["split"] == "train", "PATNO"])
    test_ids = set(manifest.loc[manifest["split"] == "test", "PATNO"])
    train_panel = panel[panel["PATNO"].isin(train_ids)]
    test_panel = panel[panel["PATNO"].isin(test_ids)]

    data = hbayes.build_growth_data(train_panel)
    say(f"  {data.n_obs:,} observations from {data.n_patients} patients")
    post = hbayes.fit_growth_model(data, n_draws=2000, burn_in=500, seed=cfg.seed)
    summary = post.summary()
    summary.to_csv(out_dir / "posterior_summary.csv", index=False)
    say("")
    say(frame(summary.round(4)))

    half = post.n_draws // 2
    say("\n  Chain stability (first half vs second half of retained draws):")
    for nm, v in [("beta0", post.beta[:, 0]), ("beta1", post.beta[:, 1]),
                  ("sigma2", post.sigma2), ("tau0^2", post.tau0_2), ("tau1^2", post.tau1_2)]:
        say(f"    {nm:8} {v[:half].mean():+.5f}  vs  {v[half:].mean():+.5f}")

    b1 = post.beta[:, 1]
    say(f"\n  Population progression: the ratio falls {-b1.mean():.4f} per year")
    say(f"  (95% credible interval {-np.quantile(b1, .975):.4f} to {-np.quantile(b1, .025):.4f}).")
    say(f"  P(population slope < 0) = {float((b1 < 0).mean()):.4f}")
    say(f"\n  Between-patient SD of progression rate: {np.sqrt(post.tau1_2).mean():.4f} per year.")
    say("  So patients differ substantially in how fast they move, which is the")
    say("  individualised trajectory the proposal is after.")

    # ------------------------------------------------ the noise problem
    section("3. WHY BAND PREDICTION IS HARD FOR THIS MODEL")
    sigma = float(np.sqrt(post.sigma2.mean()))
    band = hbayes.TD_THRESHOLD - hbayes.PIGD_THRESHOLD
    say(f"  Total residual SD                sigma = {sigma:.3f}")
    say(f"  Width of the Indeterminate band          {band:.2f}  (0.90 to 1.15)")
    say(f"  Ratio                                    {sigma/band:.1f}x")
    say("")
    say("  The model must predict which of three bands the next ratio falls in, but")
    say("  its residual spread is several times the width of the middle band. Even a")
    say("  perfectly estimated trajectory would place the next observation almost")
    say("  anywhere, so the Indeterminate class is nearly impossible to hit.")
    say("")
    say("  Note also that sigma here is the TOTAL residual SD. The proposal writes")
    say("  separate eps^obs and eps^proc, but both are additive on the same")
    say("  observation, so only their sum is identifiable; see src/ppmi_subtype/hbayes.py.")

    # --------------------------------------------------- head to head
    section("4. HEAD TO HEAD WITH THE STAGE 3 HIDDEN MARKOV MODEL")
    post_h, hmm_row, hb_row, n_common = head_to_head(
        panel, train_panel, test_panel, cfg, cfg.seed, 200 if quick else 400
    )
    say(f"  Both models fitted on the same {len(train_ids)} training patients and scored")
    say(f"  on the same {n_common} held-out transitions -- those where every visit")
    say("  involved has a finite ratio, so the growth model can score them too.\n")
    comparison = pd.DataFrame([hmm_row, hb_row]).sort_values("loglik_per_pair", ascending=False)
    comparison.to_csv(out_dir / "head_to_head.csv", index=False)
    say(frame(comparison.round(4)))

    # ------------------------------------------------------ robustness
    section("5. ROBUSTNESS ACROSS SPLITS")
    n_splits = 3 if quick else 6
    say(f"  Repeating the whole comparison on {n_splits} independent patient-level splits.\n")
    rows = []
    for sd in range(n_splits):
        man = splitting.split_patients(panel, cfg.splits["train"], seed=sd)
        tr = panel[panel["PATNO"].isin(set(man.loc[man["split"] == "train", "PATNO"]))]
        te = panel[panel["PATNO"].isin(set(man.loc[man["split"] == "test", "PATNO"]))]
        _, h_row, b_row, n_c = head_to_head(panel, tr, te, cfg, sd, 200 if quick else 300)
        rows.append({"split": sd, "n_common": n_c,
                     "HMM": h_row["loglik_per_pair"], "HB growth": b_row["loglik_per_pair"],
                     "HMM_brier": h_row["brier"], "HB_brier": b_row["brier"]})
        say(f"    split {sd}: n={n_c:4d}   HMM {h_row['loglik_per_pair']:+.4f}   "
            f"HB {b_row['loglik_per_pair']:+.4f}")
    rob = pd.DataFrame(rows)
    rob["HMM - HB"] = rob["HMM"] - rob["HB growth"]
    rob.to_csv(out_dir / "robustness.csv", index=False)
    say("")
    say(frame(rob.round(4)))
    d = rob["HMM - HB"]
    say(f"\n  HMM ahead in {int((d > 0).sum())} of {len(d)} splits, mean {d.mean():+.4f}"
        f" (sd {d.std():.4f}).")
    if len(d) >= 3:
        say(f"  Paired t-test p = {ttest_rel(rob['HMM'], rob['HB growth']).pvalue:.2e}"
            f"   Wilcoxon p = {wilcoxon(rob['HMM'], rob['HB growth']).pvalue:.4f}")

    # ------------------------------------------------ what HB gives us
    section("6. WHAT THE GROWTH MODEL PROVIDES THAT THE MARKOV MODEL DOES NOT")
    say("  A population progression rate on the clinical scale, with uncertainty:")
    say(f"    ratio declines {-b1.mean():.4f} per year "
        f"[{-np.quantile(b1,.975):.4f}, {-np.quantile(b1,.025):.4f}]")
    say("\n  A per-patient progression rate, drawn from a population distribution:")
    say(f"    between-patient SD of slope  {np.sqrt(post.tau1_2).mean():.4f} / year")
    say(f"    between-patient SD of baseline {np.sqrt(post.tau0_2).mean():.4f}")
    say("\n  Expected years for the population trajectory to cross a threshold, from")
    say(f"  the mean baseline ratio {post.beta[:,0].mean():.3f}:")
    for name, thr in (("TD threshold (1.15)", hbayes.TD_THRESHOLD),
                      ("PIGD threshold (0.90)", hbayes.PIGD_THRESHOLD)):
        years = (post.beta[:, 0] - thr) / (-b1)
        say(f"    {name:24} {np.median(years):6.1f} years "
            f"[{np.quantile(years,.025):.1f}, {np.quantile(years,.975):.1f}]")

    table = hbayes.clinical_table(post, current_ratio=float(post.beta[:, 0].mean()),
                                  horizons=tuple(cfg.model["horizons"]))
    table.to_csv(out_dir / "clinical_table_hbayes.csv", index=False)
    say("\n  Clinical table for a patient at the population baseline ratio:")
    say(frame(table.round(3)))

    section("SUMMARY")
    say(f"  Prediction of the next label : HMM ahead by {d.mean():+.4f} log-lik per")
    say(f"                                 transition, in {int((d>0).sum())}/{len(d)} splits.")
    say("  Interpretable progression    : only the growth model provides it --- a")
    say(f"                                 population slope of {-b1.mean():.4f}/year with a")
    say("                                 credible interval, and per-patient slopes.")
    say("  The two answer different questions and the project needs both.")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt and "
          f"{len(list(out_dir.glob('*.csv')))} CSV tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
