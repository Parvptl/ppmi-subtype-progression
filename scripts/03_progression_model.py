"""Stage 3 - baseline probabilistic progression model (Weeks 5-7 deliverable).

Fits and validates three progression models on held-out patients:

    Discrete Markov chain   the model the proposal originally sketched:
                            one transition matrix per visit, interval ignored
    Continuous-time chain   generator Q, P(t) = exp(Qt), interval respected
    Misclassification HMM   latent state evolving as a CTMC, observed through
                            an error matrix E

and produces the clinical output table the proposal asks for: the probability
of occupying each subtype at 1, 2, 3 and 5 years given the current state.

Requires outputs/visit_panel.csv from scripts/01_build_panel.py.

    python scripts/03_progression_model.py [--quick]

Writes to outputs/model/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import ctmc, hmm, model_plots, splitting, transitions as T  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

STATES = [1.0, 2.0, 3.0]
NAMES = ["TD", "Indeterminate", "PIGD"]

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def frame(df: pd.DataFrame, index: bool = True) -> str:
    return "    " + df.to_string(index=index).replace("\n", "\n    ")


# ---------------------------------------------------------------------------
def markov_property_test(panel: pd.DataFrame) -> pd.DataFrame:
    """Does the next state depend on the previous one, given the current one?

    Under a Markov chain in the observed label it must not. This is the test
    that motivates the hidden-state model.
    """
    df = panel[panel["subtype"].notna()].sort_values(["PATNO", "years_from_baseline"])
    g = df.groupby("PATNO", sort=False)
    trip = pd.DataFrame(
        {"s0": df["subtype"], "s1": g["subtype"].shift(-1), "s2": g["subtype"].shift(-2)}
    ).dropna()

    rows = []
    for k, s1 in enumerate(STATES):
        sub = trip[trip["s1"] == s1]
        table = pd.crosstab(sub["s0"], sub["s2"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        rows.append(
            {"current_state": NAMES[k], "n_triples": len(sub), "chi2": chi2, "dof": dof, "p_value": p}
        )
    return pd.DataFrame(rows), trip


def main() -> int:
    quick = "--quick" in sys.argv
    cfg = load_config()
    mcfg = cfg.model
    out_dir = cfg.path("outputs") / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        print(f"{panel_path} not found; run scripts/01_build_panel.py first", file=sys.stderr)
        return 1

    panel = pd.read_csv(panel_path)
    pairs = T.consecutive_pairs(panel)

    # ---------------------------------------------------------------- setup
    section("0. DATA AND SPLIT")
    say(f"  labelled visits            {panel['subtype'].notna().sum():,}")
    say(f"  transition pairs           {len(pairs):,}")
    say(f"  patients with a transition {pairs['PATNO'].nunique():,}")

    manifest = splitting.split_patients(panel, cfg.splits["train"], cfg.seed)
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    summary = splitting.split_summary(manifest, cfg.subtype.label_names)
    say("\n  Patient-level split, stratified by baseline subtype:")
    say(frame(summary))

    train_ids = set(manifest.loc[manifest["split"] == "train", "PATNO"])
    test_ids = set(manifest.loc[manifest["split"] == "test", "PATNO"])
    train_pairs = splitting.apply_split(pairs, manifest, "train")
    test_pairs = splitting.apply_split(pairs, manifest, "test")
    seq_train = hmm.build_sequences(panel[panel["PATNO"].isin(train_ids)], STATES)
    seq_test = hmm.build_sequences(panel[panel["PATNO"].isin(test_ids)], STATES)
    say(f"\n  train {len(train_pairs):,} pairs / {len(seq_train)} sequences"
        f"   test {len(test_pairs):,} pairs / {len(seq_test)} sequences")

    # ------------------------------------------------ why a hidden state
    section("1. IS THE OBSERVED PROCESS MARKOV?")
    say("  A chain in the observed label assumes P(next | current) does not depend")
    say("  on the previous state. Testing that on labelled triples:\n")
    mk, trip = markov_property_test(panel)
    mk.to_csv(out_dir / "markov_property_test.csv", index=False)
    say(frame(mk.round({"chi2": 1, "p_value": 6}), index=False))

    sub = trip[trip["s1"] == 1.0]
    tab = pd.crosstab(sub["s0"], sub["s2"], normalize="index").round(3)
    tab.index = [cfg.subtype.label_names[i] for i in tab.index]
    tab.columns = [cfg.subtype.label_names[c] for c in tab.columns]
    say("\n  Concretely, patients currently labelled TD, by where they came from:")
    say(frame(tab))
    say("\n  Coming from PIGD raises the chance of being PIGD next visit from")
    say(f"  {tab.loc['TD', 'PIGD']:.3f} to {tab.loc['PIGD', 'PIGD']:.3f}. The observed process has memory,")
    say("  so a Markov chain in the label is misspecified. That is the case for a")
    say("  latent state observed through misclassification.")

    # ----------------------------------------------------------- fit models
    section("2. FITTED MODELS (training patients only)")
    dtmc = ctmc.fit_dtmc(train_pairs, STATES)
    marginal = ctmc.fit_marginal(train_pairs, STATES)
    ct = ctmc.fit_ctmc(train_pairs, STATES, NAMES, seed=cfg.seed,
                       n_restarts=mcfg["n_restarts"])
    hm = hmm.fit_hmm(seq_train, STATES, NAMES, seed=cfg.seed,
                     n_restarts=mcfg["n_restarts"],
                     emission_floor=mcfg["emission_floor"])

    say(f"  CTMC  converged={ct.converged}  train loglik {ct.loglik:9.2f}  params {ct.n_params}")
    say(f"  HMM   converged={hm.converged}  train loglik {hm.loglik:9.2f}  params {hm.n_params}")

    say("\n  CTMC generator Q, per year (observed label treated as the state):")
    say(frame(ct.generator_frame().round(4)))
    say("\n  HMM latent generator Q, per year:")
    say(frame(hm.generator_frame().round(4)))
    say("\n  HMM misclassification matrix E (rows = true state, cols = recorded label):")
    say(frame(hm.emission_frame().round(3)))
    say(f"\n  (emission floor {hm.emission_floor:.2f}; see config model.emission_floor)")

    ct.generator_frame().to_csv(out_dir / "ctmc_generator.csv")
    hm.generator_frame().to_csv(out_dir / "hmm_latent_generator.csv")
    hm.emission_frame().to_csv(out_dir / "hmm_emission.csv")

    say("\n  Mean time spent in a state before leaving it (years):")
    sojourn = pd.DataFrame({"CTMC": ct.mean_sojourn(), "HMM (latent)": hm.mean_sojourn()})
    say(frame(sojourn.round(2)))
    sojourn.to_csv(out_dir / "mean_sojourn.csv")

    # ------------------------------------------------------- held-out test
    section("3. HELD-OUT PERFORMANCE")
    say(f"  {len(seq_test)} test patients, {len(test_pairs):,} transitions. Higher log-likelihood")
    say("  per transition is better; lower Brier is better.\n")
    results = pd.DataFrame(
        [
            hmm.evaluate(hm, seq_test, "HMM (latent + misclassification)"),
            ctmc.evaluate(ct, test_pairs, STATES, "CTMC (interval-aware)"),
            ctmc.evaluate(dtmc, test_pairs, STATES, "Discrete Markov (per-visit)"),
            ctmc.evaluate(marginal, test_pairs, STATES, "Marginal (null)"),
        ]
    ).sort_values("loglik_per_pair", ascending=False)
    results.to_csv(out_dir / "heldout_performance.csv", index=False)
    say(frame(results.round(4), index=False))

    section("4. HELD-OUT PERFORMANCE BY INTERVAL")
    rows = []
    for lo, hi, label in [(0, 1.5, "<= 1.5"), (1.5, 2.5, "1.5-2.5"), (2.5, 99, "> 2.5")]:
        sub = test_pairs[(test_pairs["interval_years"] >= lo) & (test_pairs["interval_years"] < hi)]
        if len(sub) < 10:
            continue
        rows.append(
            {
                "interval_years": label,
                "n": len(sub),
                "CTMC": ctmc.evaluate(ct, sub, STATES, "c")["loglik_per_pair"],
                "DTMC": ctmc.evaluate(dtmc, sub, STATES, "d")["loglik_per_pair"],
            }
        )
    by_interval = pd.DataFrame(rows)
    by_interval["CTMC - DTMC"] = by_interval["CTMC"] - by_interval["DTMC"]
    by_interval.to_csv(out_dir / "heldout_by_interval.csv", index=False)
    say(frame(by_interval.round(4), index=False))
    say("\n  The continuous-time chain does not beat the discrete one at any interval,")
    say("  and loses by most at long intervals -- where respecting the gap should have")
    say("  helped. Both are misspecified in the same way, per section 1.")

    # ------------------------------------------------------- robustness
    section("5. ROBUSTNESS ACROSS SPLITS")
    n_splits = 3 if quick else int(mcfg["robustness_splits"])
    say(f"  Repeating the entire procedure on {n_splits} independent patient-level splits.\n")
    rows = []
    for sd in range(n_splits):
        man = splitting.split_patients(panel, cfg.splits["train"], seed=sd)
        tr_ids = set(man.loc[man["split"] == "train", "PATNO"])
        te_ids = set(man.loc[man["split"] == "test", "PATNO"])
        tr = splitting.apply_split(pairs, man, "train")
        te = splitting.apply_split(pairs, man, "test")
        s_tr = hmm.build_sequences(panel[panel["PATNO"].isin(tr_ids)], STATES)
        s_te = hmm.build_sequences(panel[panel["PATNO"].isin(te_ids)], STATES)

        h = hmm.fit_hmm(s_tr, STATES, NAMES, seed=sd, n_restarts=2,
                        emission_floor=mcfg["emission_floor"])
        c = ctmc.fit_ctmc(tr, STATES, NAMES, seed=sd, n_restarts=2)
        rows.append(
            {
                "split": sd,
                "n_test_pairs": len(te),
                "HMM": hmm.evaluate(h, s_te)["loglik_per_pair"],
                "CTMC": ctmc.evaluate(c, te, STATES, "c")["loglik_per_pair"],
                "DTMC": ctmc.evaluate(ctmc.fit_dtmc(tr, STATES), te, STATES, "d")["loglik_per_pair"],
                "Marginal": ctmc.evaluate(ctmc.fit_marginal(tr, STATES), te, STATES, "m")["loglik_per_pair"],
            }
        )
    rob = pd.DataFrame(rows)
    rob["HMM - DTMC"] = rob["HMM"] - rob["DTMC"]
    rob.to_csv(out_dir / "robustness_across_splits.csv", index=False)
    say(frame(rob.round(4), index=False))

    diff = rob["HMM - DTMC"]
    say(f"\n  HMM beats the discrete chain in {int((diff > 0).sum())} of {len(diff)} splits.")
    say(f"  Mean advantage {diff.mean():+.4f} log-likelihood per transition (sd {diff.std():.4f}).")
    if len(diff) >= 3:
        say(f"  Paired t-test p = {ttest_rel(rob['HMM'], rob['DTMC']).pvalue:.2e}"
            f"   Wilcoxon p = {wilcoxon(rob['HMM'], rob['DTMC']).pvalue:.4f}")

    # ------------------------------------------------------- calibration
    section("6. CALIBRATION (held-out, probability of PIGD at next visit)")
    say("  A well-calibrated model puts the observed rate in each bin close to its")
    say("  mean prediction.\n")
    cal_hmm = hmm.calibration_table(hm, seq_test, target_state=3.0, n_bins=5)
    cal_hmm.to_csv(out_dir / "calibration_hmm.csv", index=False)
    say("  HMM:")
    say(frame(cal_hmm.round(3), index=False))

    cal_dtmc = ctmc.calibration_table(dtmc, test_pairs, STATES, target_state=3.0, n_bins=5)
    cal_dtmc.to_csv(out_dir / "calibration_dtmc.csv", index=False)
    say("\n  Discrete Markov chain (only three distinct predictions are possible,")
    say("  one per current state, so the curve collapses to a few bins):")
    say(frame(cal_dtmc.round(3), index=False))

    # ------------------------------------------------- clinical output
    section("7. CLINICAL OUTPUT TABLE")
    horizons = tuple(mcfg["horizons"])
    say(f"  Probability of occupying each subtype at {horizons} years, from the HMM's")
    say("  latent generator. Rows are the patient's current phenotype.\n")
    clinical = hmm.clinical_table(hm, horizons=horizons)
    clinical.to_csv(out_dir / "clinical_table_hmm.csv", index=False)
    for state in NAMES:
        block = clinical[clinical["current_state"] == state].drop(columns="current_state")
        say(f"  Current subtype: {state}")
        say(frame(block.set_index("horizon_years").round(3)))
        say("")

    ctmc.clinical_table(ct, horizons=horizons).to_csv(
        out_dir / "clinical_table_ctmc.csv", index=False
    )

    # ------------------------------------------------------- bootstrap
    if not quick:
        section("8. BOOTSTRAP CONFIDENCE INTERVALS (CTMC rates)")
        n_boot = int(mcfg["bootstrap_draws"])
        say(f"  {n_boot} patient-level resamples of the training set.\n")
        boot = ctmc.bootstrap_rates(train_pairs, STATES, NAMES, n_boot=n_boot, seed=cfg.seed)
        fitted = [
            {"from_state": NAMES[a], "to_state": NAMES[b], "rate": ct.Q[a, b]}
            for a in range(3)
            for b in range(3)
            if a != b
        ]
        boot = pd.DataFrame(fitted).merge(boot, on=["from_state", "to_state"])
        boot.to_csv(out_dir / "ctmc_rate_bootstrap.csv", index=False)
        say(frame(boot.round(4), index=False))

    # --------------------------------------------------------- figures
    section("9. FIGURES")
    fig_dir = out_dir / "figures"
    written = [
        model_plots.progression_curves(hm, fig_dir / "progression_curves.png",
                                       horizons=horizons),
        model_plots.model_comparison(rob, fig_dir / "model_comparison.png"),
        model_plots.emission_heatmap(hm.emission_frame(), fig_dir / "misclassification.png"),
        model_plots.sojourn_comparison(sojourn, fig_dir / "sojourn_comparison.png"),
        model_plots.calibration_plot(cal_hmm, fig_dir / "calibration_hmm.png",
                                     "HMM calibration: probability of PIGD at next visit"),
    ]
    for p in written:
        say(f"  {p.relative_to(cfg.root)}")

    section("SUMMARY")
    best = results.iloc[0]
    say(f"  Best held-out model: {best['model']}")
    say(f"    log-likelihood per transition {best['loglik_per_pair']:.4f}")
    say(f"    accuracy {best['accuracy']:.3f}   Brier {best['brier']:.4f}")
    say(f"  Beats the discrete baseline in {int((diff > 0).sum())}/{len(diff)} independent splits.")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt "
          f"and {len(list(out_dir.glob('*.csv')))} CSV tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
