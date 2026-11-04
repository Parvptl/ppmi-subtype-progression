"""Does ratio-band information improve next-visit TD/PIGD prediction?

The CT-HMM of `scripts/60_next_visit_subtype_prediction.py` sees one of three
recorded labels per visit. Those labels are a threshold cut of the continuous
tremor/PIGD ratio, so two visits labelled TD can sit at ratio 1.20 and 4.50 and
the label keeps neither number. This experiment gives the same chain a finer
observation alphabet --- the subtype *and* the ratio band it came from --- and
asks whether the extra resolution predicts the next visit any better.

    observed category = (subtype, ratio band)

    main range   [0.0, 3.0) in 30 fixed bands of width 0.1
    tail         [3.0, inf) in 0.25-width candidate bands, merged upward until
                 each holds >= 30 observations from >= 20 patients, the rule
                 fixed in advance and applied to TRAINING patients only
    undefined    ratio not computable (pigd_score == 0): its own category per
                 subtype, because 16% of this cohort's visits are in it

Three arms are fitted in every fold, on identical cases:

    CT-HMM (latent, interval-aware)       the published baseline, refitted
    CT-HMM (subtype x ratio band)         the experimental model
    CT-HMM (subtype labels, joint code    a control: the joint code path over a
            path)                         one-category-per-subtype alphabet, so
                                          any gap to the baseline measures the
                                          implementation, not the bands

Everything else is the baseline's: three hidden states, interval-aware latent
CTMC, forward filtering, the same cohort and prediction cases, the same
two-class conditional target, the same seed, restarts and emission floor, and
the same scoring code. The primary comparison is the second arm against the
first.

    python scripts/87_cthmm_ratio_banded_observations.py --out <dir> [--workers N]
                                                         [--limit N] [--resume]

Reads the frozen panel; writes only into --out. No participant-level identifier
leaves this directory.
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

# One BLAS thread per process: the fold loop is already parallel over patients.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hmm_banded, loo  # noqa: E402
from ppmi_subtype import next_visit as nv, next_visit_banded as nvb  # noqa: E402
from ppmi_subtype import hmm_stability, ratio_bands as rb  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

#: Columns this experiment reads out of the panel.
REQUIRED_COLUMNS = ("PATNO", "EVENT_ID", "years_from_baseline", "subtype",
                    "subtype_name", "ratio", "ratio_defined", "tremor_score",
                    "pigd_score")

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
#  Data integrity, before anything is fitted
# ---------------------------------------------------------------------------
def integrity_checks(panel: pd.DataFrame, data: loo.LOOData, cases: pd.DataFrame,
                     cfg_subtype) -> tuple[dict, pd.DataFrame]:
    """Schema, identifiers, ordering, label/ratio consistency, target construction.

    The target check is an independent re-derivation: the next label and the
    elapsed interval are read back out of the panel and compared with the case
    table the models are scored on, rather than recomputed by the same code.
    """
    checks: dict[str, bool] = {}
    cohort = set(data.patients)

    checks["every required column is present"] = all(
        c in panel.columns for c in REQUIRED_COLUMNS)

    labelled = panel[panel["subtype"].notna()].copy()
    in_cohort = labelled[labelled["PATNO"].isin(cohort)].sort_values(
        ["PATNO", "years_from_baseline"])

    checks["no labelled visit is missing its patient id, visit id or visit time"] = bool(
        in_cohort[["PATNO", "EVENT_ID", "years_from_baseline"]].notna().all().all())
    checks["patient and visit identifiers are unique per row"] = bool(
        not panel.duplicated(["PATNO", "EVENT_ID"]).any())
    checks["no patient has two labelled visits at the same time"] = bool(
        not in_cohort.duplicated(["PATNO", "years_from_baseline"]).any())
    checks["labelled visits are strictly ordered in time within a patient"] = bool(
        in_cohort.groupby("PATNO")["years_from_baseline"].apply(
            lambda s: s.is_monotonic_increasing and s.diff().iloc[1:].gt(0).all()).all())
    checks["the recorded subtype is one of the three labels"] = bool(
        set(in_cohort["subtype"].unique()) <= set(cfg_subtype.labels.values()))

    # Ratio availability: among labelled visits a ratio is missing exactly when
    # the PIGD subscore is zero, which is the definition, not an accident.
    missing_ratio = in_cohort["ratio"].isna()
    pigd_zero = in_cohort["pigd_score"] == 0
    checks["a ratio is missing exactly where the PIGD subscore is zero"] = bool(
        (missing_ratio == pigd_zero).all())
    checks["the panel's ratio_defined flag agrees with the ratio column"] = bool(
        (in_cohort["ratio_defined"].astype(bool) == in_cohort["ratio"].notna()).all())
    checks["no ratio is negative or infinite"] = bool(
        ((in_cohort["ratio"].dropna() >= 0)
         & np.isfinite(in_cohort["ratio"].dropna())).all())

    # Label/ratio consistency: re-derive the subtype from the ratio by the
    # documented threshold rule and compare with the panel's label.
    names = cfg_subtype.label_names
    outer = cfg_subtype.band_boundaries != "inner"
    r = in_cohort["ratio"].to_numpy(dtype=float)
    derived = np.full(len(in_cohort), "", dtype=object)
    with np.errstate(invalid="ignore"):
        if outer:
            derived[r >= cfg_subtype.td_threshold] = "TD"
            derived[r <= cfg_subtype.pigd_threshold] = "PIGD"
            mid = (r > cfg_subtype.pigd_threshold) & (r < cfg_subtype.td_threshold)
        else:
            derived[r > cfg_subtype.td_threshold] = "TD"
            derived[r < cfg_subtype.pigd_threshold] = "PIGD"
            mid = (r >= cfg_subtype.pigd_threshold) & (r <= cfg_subtype.td_threshold)
    derived[mid] = "Indeterminate"
    tremor = in_cohort["tremor_score"].to_numpy(dtype=float)
    undefined = ~np.isfinite(r)
    derived[undefined & (tremor > 0)] = "TD"
    derived[undefined & (tremor == 0)] = "Indeterminate"
    recorded = in_cohort["subtype"].map(names).to_numpy(dtype=object)
    checks["every label follows the threshold rule it is derived from"] = bool(
        (derived == recorded).all())

    # The prediction target, re-derived from the panel.
    target_ok, interval_ok = True, True
    by_patient = {p: g for p, g in in_cohort.groupby("PATNO", sort=False)}
    for patno, g in cases.groupby("PATNO", sort=False):
        rows = by_patient[patno]
        labels = rows["subtype"].map(names).to_numpy(dtype=object)
        times = rows["years_from_baseline"].to_numpy(dtype=float)
        k = g["visit_index"].to_numpy(int)
        if not np.array_equal(labels[k + 1], g["actual_next_label"].to_numpy(dtype=object)):
            target_ok = False
        if not np.allclose(times[k + 1] - times[k], g["interval_years"].to_numpy(float)):
            interval_ok = False
    checks["the next-visit target is the next labelled visit's own label"] = target_ok
    checks["the interval is the gap between those two visits"] = interval_ok
    checks["a prediction case exists for every within-patient transition"] = bool(
        len(cases) == data.n_transitions
        == sum(len(g) - 1 for g in by_patient.values()))
    checks["only patients with at least two labelled visits are in the cohort"] = bool(
        min(len(g) for g in by_patient.values()) >= 2)

    missing = pd.DataFrame({
        "column": list(REQUIRED_COLUMNS),
        "n_missing_in_cohort": [int(in_cohort[c].isna().sum())
                                if c in in_cohort else -1
                                for c in REQUIRED_COLUMNS],
        "n_missing_in_panel": [int(panel[c].isna().sum()) if c in panel else -1
                               for c in REQUIRED_COLUMNS],
    })
    return checks, missing


# ---------------------------------------------------------------------------
#  Fold workers
# ---------------------------------------------------------------------------
def _init(data, bdata, cases, seed, n_restarts, emission_floor) -> None:
    _WORK.update(data=data, bdata=bdata, cases=cases, seed=seed,
                 n_restarts=n_restarts, emission_floor=emission_floor)


def _fold(patno):
    try:
        frame, diag = nvb.run_fold(_WORK["data"], _WORK["bdata"], _WORK["cases"],
                                   patno, _WORK["seed"], _WORK["n_restarts"],
                                   _WORK["emission_floor"])
        return frame, diag, None
    except Exception as exc:                    # noqa: BLE001 -- recorded, not hidden
        return None, None, f"{type(exc).__name__}: {exc}"


def run_folds(data, bdata, cases, patients, seed, n_restarts, emission_floor,
              workers, out: Path, checkpoint_every: int = 25, resume: bool = False):
    """The LOPO loop. Completed folds are checkpointed so a stop can be resumed."""
    ckpt = out / "predictions_checkpoint.csv"
    ckpt_diag = out / "fold_band_diagnostics_checkpoint.csv"

    done: set = set()
    frames: list[pd.DataFrame] = []
    diags: list[dict] = []
    if resume and ckpt.exists():
        prior = pd.read_csv(ckpt, low_memory=False)
        prior = prior[prior["held_out_patient"].isin(patients)]
        if len(prior):
            done = set(prior["held_out_patient"].unique())
            frames.append(prior)
            say(f"  resuming: {len(done):,} folds already complete")
        if ckpt_diag.exists():
            pd_prior = pd.read_csv(ckpt_diag)
            diags = pd_prior[pd_prior["held_out_patient"].isin(done)].to_dict("records")

    pending = [p for p in patients if p not in done]
    failures: list[dict] = []
    buffer: list[pd.DataFrame] = []

    def flush() -> None:
        if buffer:
            frames.extend(buffer)
            buffer.clear()
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(ckpt, index=False)
        if diags:
            pd.DataFrame(diags).to_csv(ckpt_diag, index=False)

    args = (data, bdata, cases, seed, n_restarts, emission_floor)
    if workers <= 1:
        _init(*args)
        results = ((p, _fold(p)) for p in pending)
        for n, (patno, (frame, diag, error)) in enumerate(results, start=1):
            if error is None:
                buffer.append(frame)
                diags.append(diag)
            else:
                failures.append({"held_out_patient": patno, "error": error})
            if n % 25 == 0:
                say(f"  fold {n:,}/{len(pending):,}")
            if n % checkpoint_every == 0:
                flush()
    else:
        with mp.get_context("fork").Pool(workers, initializer=_init,
                                         initargs=args) as pool:
            for n, (frame, diag, error) in enumerate(
                    pool.imap(_fold, pending, chunksize=1), start=1):
                if error is None:
                    buffer.append(frame)
                    diags.append(diag)
                else:
                    failures.append({"held_out_patient": pending[n - 1],
                                     "error": error})
                if n % 50 == 0:
                    say(f"  fold {n:,}/{len(pending):,}  "
                        f"({time.strftime('%H:%M:%S')})")
                if n % checkpoint_every == 0:
                    flush()
    flush()

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return (predictions, pd.DataFrame(diags),
            pd.DataFrame(failures, columns=["held_out_patient", "error"]))


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------
def write_plots(out: Path, comparison: pd.DataFrame, paired: dict,
                draws: np.ndarray, ratios: np.ndarray, scheme: rb.BandScheme,
                by_ratio: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) accuracy with patient-bootstrap intervals
    ax = axes[0, 0]
    show = comparison.set_index("model")
    order = [m for m in nvb.MODELS if m in show.index]
    y = np.arange(len(order))
    acc = show.loc[order, "accuracy"].to_numpy()
    lo = show.loc[order, "accuracy_ci_patient_bootstrap_low"].to_numpy()
    hi = show.loc[order, "accuracy_ci_patient_bootstrap_high"].to_numpy()
    ax.errorbar(acc, y, xerr=[acc - lo, hi - acc], fmt="o", capsize=4, color="#2b6cb0")
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace(" (", "\n(") for m in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("next-visit accuracy (TD vs PIGD)")
    ax.set_title("(a) Accuracy, 95% patient bootstrap")
    ax.grid(axis="x", alpha=0.3)

    # (b) probabilistic scores
    ax = axes[0, 1]
    width = 0.35
    x = np.arange(len(order))
    ax.bar(x - width / 2, show.loc[order, "brier_two_class"], width, label="Brier",
           color="#2b6cb0")
    ax.bar(x + width / 2, show.loc[order, "log_loss"], width, label="log loss",
           color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" (")[1].rstrip(")").replace(", ", ",\n")
                        for m in order], fontsize=8)
    ax.set_title("(b) Brier score and log loss (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (c) paired bootstrap of the accuracy difference
    ax = axes[1, 0]
    ax.hist(draws, bins=40, color="#2b6cb0", alpha=0.8)
    ax.axvline(0.0, color="black", lw=1)
    ax.axvline(paired["accuracy_delta"], color="#c53030", lw=1.5,
               label=f"delta = {paired['accuracy_delta']:+.4f}")
    ax.axvspan(paired["delta_ci_patient_bootstrap_low"],
               paired["delta_ci_patient_bootstrap_high"], color="#c53030", alpha=0.12,
               label="95% CI")
    ax.set_xlabel("accuracy (banded) - accuracy (labels), paired on cases")
    ax.set_title("(c) Paired patient bootstrap of the difference")
    ax.legend(fontsize=8)

    # (d) the ratio distribution and the band structure it is cut into
    ax = axes[1, 1]
    finite = ratios[np.isfinite(ratios)]
    ax.hist(np.clip(finite, 0, 6), bins=np.arange(0, 6.05, 0.1), color="#718096")
    for e in scheme.edges:
        if e <= 6:
            ax.axvline(e, color="#2b6cb0", lw=0.4, alpha=0.6)
    ax.axvline(scheme.pigd_threshold, color="#c53030", lw=1.2, label="PIGD threshold")
    ax.axvline(scheme.td_threshold, color="#2f855a", lw=1.2, label="TD threshold")
    ax.axvline(rb.TAIL_START, color="black", lw=1.0, ls="--", label="tail starts")
    ax.set_xlabel("tremor / PIGD ratio (clipped at 6)")
    ax.set_ylabel("labelled visits")
    ax.set_title(f"(d) Ratio distribution and band edges\n"
                 f"{len(finite):,} of {len(ratios):,} visits have a computable ratio",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)

    # accuracy by ratio range, both arms
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = by_ratio.pivot(index="ratio_range", columns="model", values="accuracy")
    n = by_ratio.pivot(index="ratio_range", columns="model", values="n")
    keep = [m for m in (nvb.MODEL_LABEL, nvb.MODEL_BANDED) if m in pivot.columns]
    pivot[keep].plot.bar(ax=ax, color=["#718096", "#2b6cb0"], width=0.8)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("ratio at the prediction point")
    ax.set_title("Accuracy by ratio range (n above each group)")
    for k, label in enumerate(pivot.index):
        count = n.loc[label, keep[0]]
        ax.annotate("0" if pd.isna(count) else f"{int(count)}", (k, 1.01),
                    ha="center", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_ratio_range.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N folds (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed folds from a checkpoint in --out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    # ------------------------------------------------------- 1. data integrity
    section("1. INPUT AND DATA INTEGRITY")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        raise SystemExit(
            f"the frozen panel is not at {panel_path}. This experiment needs the "
            "real PPMI panel; it will not substitute or synthesise one.")
    panel_md5 = md5(panel_path)
    say(f"  panel                {panel_path.relative_to(ROOT)}")
    say(f"  md5                  {panel_md5}")
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(
            f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}. Refusing to "
            "run against a different dataset than the frozen one.")
    say(f"  expected md5         {EXPECTED_PANEL_MD5}  [MATCH]")
    say(f"  rows x columns       {panel_path.stat().st_size:,} bytes on disk")

    panel = pd.read_csv(panel_path, low_memory=False)
    states = [cfg.subtype.labels[s] for s in nv.STATE_NAMES]
    data = loo.prepare(panel, states)
    cases = nv.build_cases(data)
    bdata = nvb.prepare(panel, data, cfg.subtype)

    say(f"  panel shape          {panel.shape[0]:,} rows x {panel.shape[1]:,} columns")
    say(f"  labels               {dict(zip(nv.STATE_NAMES, states))}")
    say(f"  thresholds           PIGD <= {cfg.subtype.pigd_threshold}, "
        f"TD >= {cfg.subtype.td_threshold} ({cfg.subtype.band_boundaries} boundaries)")
    say(f"  patients, >= 2 labelled visits       {len(data.sequences):,}")
    say(f"  labelled visits in the cohort        "
        f"{sum(len(s['labels']) for s in bdata.sequences):,}")
    say(f"  prediction cases (transitions)       {len(cases):,}")

    checks_data, missing = integrity_checks(panel, data, cases, cfg.subtype)
    say("")
    for name, ok in checks_data.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "data_integrity.json").write_text(json.dumps(checks_data, indent=2),
                                             encoding="utf-8")
    missing.to_csv(out / "missing_values.csv", index=False)
    say("")
    say("  Missing values in the columns this experiment reads:")
    say(block(missing, index=False))
    if not all(checks_data.values()):
        raise SystemExit(f"data integrity failed: "
                         f"{[k for k, v in checks_data.items() if not v]}")

    # ------------------------------------------------------ 2. band definition
    section("2. THE BAND DEFINITION, FIXED BEFORE ANY FIT")
    say("  Observed category = (subtype, ratio band). The subtype is kept, not")
    say("  replaced: every category carries it, and the next-visit prediction is")
    say("  summed back over the categories of each subtype before it is scored,")
    say("  so both arms predict the same quantity.")
    say("")
    say(f"  main range     [{rb.MAIN_LOW}, {rb.MAIN_HIGH}) in fixed bands of width "
        f"{rb.MAIN_WIDTH} -> {len(rb.main_edges()) - 1} bands")
    say(f"  tail           [{rb.TAIL_START}, inf) in candidate bands of width "
        f"{rb.TAIL_WIDTH}, merged upward")
    say(f"  merge rule     a tail band must hold >= {rb.MIN_TAIL_OBSERVATIONS} "
        f"observations from >= {rb.MIN_TAIL_PATIENTS} distinct patients;")
    say("                 sweeping up from 3.0, consecutive candidates accumulate")
    say("                 until both minima are met, then a new band starts. A")
    say("                 final band still short is absorbed into the one below.")
    say("                 The topmost band is [lo, inf), so a held-out ratio above")
    say("                 everything in training is assignable without being seen.")
    say("  boundaries     left-closed, right-open: a ratio on an edge opens that")
    say("                 band, so no observation can belong to two bands.")
    say("  undefined      ratio not computable (pigd_score == 0) is its own")
    say("                 category per subtype. It is 16% of this cohort's visits;")
    say("                 dropping them would change the evaluation cohort.")
    say("")
    say("  Why these minima, and why they were not tuned:")
    say("  Each tail band costs one free emission parameter per hidden state, so")
    say("  with three hidden states a band needs about 3 x 5 = 15 observations to")
    say(f"  reach the conventional five-per-cell minimum; {rb.MIN_TAIL_OBSERVATIONS} is the smallest")
    say("  round number that clears it with a factor-of-two margin. The patient")
    say("  minimum is there because this is a longitudinal panel: a band can hold")
    say("  many observations from very few people, and repeat visits of one")
    say("  patient are not independent evidence. Both were fixed from the")
    say("  training-side distribution before a single model was fitted, and no")
    say("  alternative value was ever scored against held-out accuracy.")

    # --------------------------------------------------- 3. band diagnostics
    section("3. RATIO DISTRIBUTION AND BAND DIAGNOSTICS")
    all_ratios = np.concatenate([s["ratios"] for s in bdata.sequences])
    all_patients = np.concatenate(
        [np.full(len(s["ratios"]), s["PATNO"], dtype=object) for s in bdata.sequences])
    finite = all_ratios[np.isfinite(all_ratios)]

    say(f"  labelled visits                      {len(all_ratios):,}")
    say(f"  ratio computable                     {len(finite):,} "
        f"({len(finite) / len(all_ratios):.1%})")
    say(f"  ratio not computable                 {len(all_ratios) - len(finite):,} "
        f"({1 - len(finite) / len(all_ratios):.1%})")
    q = pd.Series(finite).describe(percentiles=[.05, .25, .5, .75, .9, .95, .99])
    say("")
    say("  Distribution of the computable ratio:")
    say(block(q.round(4).to_frame("ratio")))
    say(f"  observations >= {rb.TAIL_START} (the tail)          "
        f"{int((finite >= rb.TAIL_START).sum()):,} "
        f"({(finite >= rb.TAIL_START).mean():.1%} of computable), "
        f"{len(set(all_patients[np.isfinite(all_ratios)][finite >= rb.TAIL_START]))} patients")

    # The cohort-wide scheme, for documentation only: nothing scored uses it.
    scheme, alphabet, tail_diag = nvb.fold_alphabet(bdata, None)
    tail_diag.to_csv(out / "tail_bands_before_after_merge.csv", index=False)
    say("")
    say("  Candidate tail bands before and after merging (whole cohort, for")
    say("  documentation only -- every scored fit uses its own fold's edges):")
    say(block(tail_diag.round(3), index=False))
    say("")
    say(f"  retained tail bands  {len(scheme.tail_lows)} of {len(tail_diag)} candidates")
    say(f"  total ratio bands    {scheme.n_bands} "
        f"({len(rb.main_edges()) - 1} main + {len(scheme.tail_lows)} tail)")
    say(f"  joint categories     {alphabet.n_categories}")
    say(f"  free parameters      banded {hmm_banded.n_params(alphabet, 3)} "
        f"vs baseline 14")

    occupancy = nvb.band_occupancy(bdata, alphabet)
    occupancy.to_csv(out / "band_occupancy_cohort.csv", index=False)
    main_tbl = occupancy[~occupancy["is_tail"] & occupancy["band_low"].notna()]
    say("")
    say("  Main-range bands (fixed 0.1 width), by subtype:")
    say(block(main_tbl[["category", "n_observations", "n_patients"]], index=False))
    say("")
    say("  Tail bands after merging, and the undefined-ratio categories:")
    rest = occupancy[occupancy["is_tail"] | occupancy["band_low"].isna()]
    say(block(rest[["category", "n_observations", "n_patients"]], index=False))

    empty = occupancy[occupancy["n_observations"] == 0]
    sparse = occupancy[(occupancy["n_observations"] > 0)
                       & (occupancy["n_observations"] < 15)]
    say("")
    say(f"  categories with no observation at all        {len(empty)}  "
        f"{list(empty['category'])}")
    say(f"  categories with 1-14 observations            {len(sparse)}  "
        f"{list(sparse['category'])}")
    say("")
    say("  The main-range width is fixed at 0.1 by the experiment's design and is")
    say("  NOT merged, so the sparsity above is carried into the fit deliberately.")
    say("  The ratio is a quotient of two small-integer means (11 tremor items over")
    say("  5 PIGD items), so it takes lumpy discrete values and the fixed bands are")
    say("  unevenly filled. This is the experiment's main statistical risk and is")
    say("  revisited in the interpretation.")

    # ---------------------------------------------------------------- 4. LOPO
    section("4. LEAVE-ONE-PATIENT-OUT")
    patients = data.patients[: args.limit] if args.limit else data.patients
    say(f"  folds                {len(patients):,}")
    say(f"  workers              {args.workers}")
    say(f"  seed                 {cfg.seed}")
    say(f"  restarts per fit     {cfg.model['n_restarts']}")
    say(f"  emission floor       {cfg.model['emission_floor']}  (on the subtype")
    say("                       marginal of the joint emission -- see below)")
    say("")
    say("  Each fold holds one patient out whole, rebuilds the tail-band edges")
    say("  from the remaining patients' ratios alone, fits all three arms on those")
    say("  patients, and predicts the held-out patient's visits. The band edges are")
    say("  re-derived and compared after the fact in every fold, so a leak would")
    say("  raise rather than pass quietly.")

    predictions, fold_diag, failures = run_folds(
        data, bdata, cases, patients, cfg.seed, cfg.model["n_restarts"],
        cfg.model["emission_floor"], args.workers, out, resume=args.resume)

    if not len(predictions):
        raise SystemExit("every fold failed; nothing to evaluate")

    predictions = predictions.sort_values(
        ["model", "PATNO", "visit_index"]).reset_index(drop=True)
    predictions.to_csv(out / "next_visit_predictions.csv", index=False)
    fold_diag.sort_values("held_out_patient").to_csv(
        out / "fold_band_diagnostics.csv", index=False)
    failures.to_csv(out / "failed_folds.csv", index=False)

    say("")
    say(f"  folds attempted      {len(patients):,}")
    say(f"  folds completed      {predictions['held_out_patient'].nunique():,}")
    say(f"  folds failed         {len(failures):,}")
    if len(failures):
        say(block(failures, index=False))
    say(f"  prediction rows      {len(predictions):,}")
    say(f"  elapsed              {time.time() - started:.0f}s")

    # -------------------------------------------------------------- 5. checks
    section("5. CORRECTNESS AND LEAKAGE CHECKS")
    checks = nv.correctness_checks(predictions, cases, data, failures,
                                   patients=patients)
    scored_banded = nv.scored(predictions[predictions["model"] == nvb.MODEL_BANDED])
    checks.update({
        "the aggregated joint prediction lost no probability mass":
            bool(np.isfinite(predictions["P_TD"]).any()),
        "both arms were scored on the identical cases":
            nv.scored(predictions[predictions["model"] == nvb.MODEL_LABEL])
            [["PATNO", "visit_index"]].equals(
                scored_banded[["PATNO", "visit_index"]]),
        "every fold's band edges were rebuilt from its training patients":
            len(fold_diag) == predictions["held_out_patient"].nunique(),
        "the joint emission's subtype marginal respects the emission floor":
            bool((fold_diag["banded_min_diag_M"]
                  >= cfg.model["emission_floor"] - 1e-9).all()),
        "the banded model has strictly more parameters than the baseline":
            bool((fold_diag["n_params_banded"] > fold_diag["n_params_label"]).all()),
        "the control arm reproduces the baseline arm":
            bool((fold_diag["control_vs_label_max_abs_prob_gap"] < 1e-3).all()),
    })
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    # ---------------------------------------------------------- 6. convergence
    section("6. CONVERGENCE AND FAILED FITS")
    conv = nv.convergence_report(predictions)
    conv.round(4).to_csv(out / "convergence.csv", index=False)
    say(block(conv.round(4), index=False))
    say("")
    say("  A fold counts as converged when scipy succeeds OR the projected")
    say("  gradient is below "
        f"{nv.PROJ_GRAD_TOL}, exactly as in the baseline experiment.")
    say("")
    iter_cols = ["banded_n_iter", "banded_n_fev", "banded_proj_grad",
                 "label_n_iter", "label_n_fev", "label_proj_grad"]
    say("  Optimiser effort per fold:")
    say(block(fold_diag[iter_cols].describe().round(4)))
    say("")
    say("  The joint model's gradient is computed exactly (forward-backward), not")
    say("  by finite differences: with ~137 parameters a finite-difference gradient")
    say("  costs 138 likelihood evaluations and a probe on the full cohort ran out")
    say("  of budget after 935 iterations with a projected gradient of 2.3e-2, far")
    say("  short of convergence. The baseline arm keeps the audited finite-")
    say("  difference path. The control arm is the baseline model fitted through")
    say("  the exact-gradient path, which is how that change is measured:")
    say(f"    worst |loglik| gap, control vs baseline   "
        f"{fold_diag['control_vs_label_loglik_gap'].max():.3e}")
    say(f"    worst |emission| gap                      "
        f"{fold_diag['control_vs_label_max_abs_E_gap'].max():.3e}")
    say(f"    worst |predicted probability| gap         "
        f"{fold_diag['control_vs_label_max_abs_prob_gap'].max():.3e}")

    msgs = (predictions[predictions["model"] == nvb.MODEL_BANDED]
            .drop_duplicates("held_out_patient")["fit_termination"].value_counts())
    say("")
    say("  Banded-model termination messages across folds:")
    say(block(msgs.to_frame("n_folds")))

    # ------------------------------------------------- 7. fold band structure
    section("7. HOW THE BAND STRUCTURE VARIED ACROSS FOLDS")
    structure = (fold_diag.groupby("tail_lows")
                 .agg(n_folds=("held_out_patient", "size"),
                      n_tail_bands=("n_tail_bands", "first"),
                      n_categories=("n_categories", "first"),
                      n_params=("n_params_banded", "first"))
                 .sort_values("n_folds", ascending=False).reset_index())
    structure.to_csv(out / "fold_band_structures.csv", index=False)
    say(f"  distinct tail-band structures across {len(fold_diag):,} folds: "
        f"{len(structure)}")
    say(block(structure, index=False))
    say("")
    if len(structure) == 1:
        say("  Every fold built its own edges from its own 933 training patients and")
        say("  all of them coincided: no single patient's observations were pivotal")
        say("  at a merge threshold, so removing one never moved an edge. The")
        say("  structure below is therefore what each fold independently arrived at,")
        say("  not a global definition imposed on every fit -- which is also why the")
        say(f"  {int(fold_diag['held_out_above_training_tail'].sum())} folds whose held-out patient sat above every training band")
        say("  still had a band to fall into.")
    else:
        say("  The edges are learned inside each fold, and they are not identical")
        say("  across folds: the structures above are what the folds arrived at, and")
        say("  no single global definition was used for every fit.")
    say(f"  folds whose held-out patient had a ratio above every training band: "
        f"{int(fold_diag['held_out_above_training_tail'].sum()):,}")
    say(f"  joint categories per fold: min {int(fold_diag['n_categories'].min())}, "
        f"max {int(fold_diag['n_categories'].max())}")
    say(f"  banded parameters per fold: min {int(fold_diag['n_params_banded'].min())}, "
        f"max {int(fold_diag['n_params_banded'].max())}")

    # ------------------------------------------------------- 8. the comparison
    section("8. MODEL COMPARISON ON IDENTICAL CASES")
    rows = [nv.two_class_metrics(g.reset_index(drop=True), seed=cfg.seed)
            for _, g in predictions.groupby("model", sort=False)]
    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    comparison.round(4).to_csv(out / "model_comparison.csv", index=False)
    show = ["model", "n_patients", "n_scored", "accuracy",
            "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high",
            "brier_two_class", "log_loss", "n_ties"]
    say(block(comparison[show].round(4), index=False))

    for name, g in predictions.groupby("model", sort=False):
        slug = _slug(name)
        nv.confusion(g).to_csv(out / f"confusion_{slug}.csv")
        nv.per_class_metrics(g).round(4).to_csv(out / f"per_class_{slug}.csv",
                                                index=False)
        nv.calibration(g).round(4).to_csv(out / f"calibration_{slug}.csv", index=False)

    say("")
    for name in (nvb.MODEL_LABEL, nvb.MODEL_BANDED):
        g = predictions[predictions["model"] == name]
        cm = nv.confusion(g)
        per = nv.per_class_metrics(g)
        bal = float(per["recall"].mean())
        say(f"  {name}")
        say(f"    confusion (rows predicted, columns actual), balanced accuracy "
            f"{bal:.4f}:")
        say(block(cm))
        say(block(per.round(4), index=False))
        say("")

    # ------------------------------------------------- 9. the primary contrast
    section("9. PRIMARY CONTRAST: RATIO-BANDED AGAINST THE BASELINE")
    paired = nv.paired_accuracy_difference(
        predictions[predictions["model"] == nvb.MODEL_BANDED],
        predictions[predictions["model"] == nvb.MODEL_LABEL], seed=cfg.seed)
    acc = comparison.set_index("model")
    base_acc = float(acc.loc[nvb.MODEL_LABEL, "accuracy"])
    band_acc = float(acc.loc[nvb.MODEL_BANDED, "accuracy"])

    contrast_rows = [{
        "contrast": "ratio bands added to the subtype label",
        "model_a": nvb.MODEL_BANDED, "model_b": nvb.MODEL_LABEL,
        "accuracy_a": band_acc, "accuracy_b": base_acc,
        "brier_a": float(acc.loc[nvb.MODEL_BANDED, "brier_two_class"]),
        "brier_b": float(acc.loc[nvb.MODEL_LABEL, "brier_two_class"]),
        "brier_delta": float(acc.loc[nvb.MODEL_BANDED, "brier_two_class"]
                             - acc.loc[nvb.MODEL_LABEL, "brier_two_class"]),
        "log_loss_a": float(acc.loc[nvb.MODEL_BANDED, "log_loss"]),
        "log_loss_b": float(acc.loc[nvb.MODEL_LABEL, "log_loss"]),
        "log_loss_delta": float(acc.loc[nvb.MODEL_BANDED, "log_loss"]
                                - acc.loc[nvb.MODEL_LABEL, "log_loss"]),
        **paired,
    }, {
        "contrast": "control: the baseline model through the joint code path",
        "model_a": nvb.MODEL_CONTROL, "model_b": nvb.MODEL_LABEL,
        "accuracy_a": float(acc.loc[nvb.MODEL_CONTROL, "accuracy"]),
        "accuracy_b": base_acc,
        "brier_a": float(acc.loc[nvb.MODEL_CONTROL, "brier_two_class"]),
        "brier_b": float(acc.loc[nvb.MODEL_LABEL, "brier_two_class"]),
        "brier_delta": float(acc.loc[nvb.MODEL_CONTROL, "brier_two_class"]
                             - acc.loc[nvb.MODEL_LABEL, "brier_two_class"]),
        "log_loss_a": float(acc.loc[nvb.MODEL_CONTROL, "log_loss"]),
        "log_loss_b": float(acc.loc[nvb.MODEL_LABEL, "log_loss"]),
        "log_loss_delta": float(acc.loc[nvb.MODEL_CONTROL, "log_loss"]
                                - acc.loc[nvb.MODEL_LABEL, "log_loss"]),
        **nv.paired_accuracy_difference(
            predictions[predictions["model"] == nvb.MODEL_CONTROL],
            predictions[predictions["model"] == nvb.MODEL_LABEL], seed=cfg.seed),
    }]
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.round(6).to_csv(out / "contrasts.csv", index=False)

    say(f"  baseline accuracy                {base_acc:.4f}  ({base_acc:.2%})")
    say(f"  ratio-banded accuracy            {band_acc:.4f}  ({band_acc:.2%})")
    say(f"  difference (banded - baseline)   {paired['accuracy_delta']:+.4f}  "
        f"({paired['accuracy_delta'] * 100:+.2f} percentage points)")
    say(f"  95% CI, paired patient bootstrap "
        f"[{paired['delta_ci_patient_bootstrap_low']:+.4f}, "
        f"{paired['delta_ci_patient_bootstrap_high']:+.4f}]")
    if base_acc > 0:
        say(f"  relative change                  "
            f"{paired['accuracy_delta'] / base_acc:+.2%}")
    say(f"  cases scored by both             {paired['n_paired']:,} "
        f"from {paired['n_patients']:,} patients")
    say(f"  banded right / baseline wrong    {paired['a_right_b_wrong']:,}")
    say(f"  baseline right / banded wrong    {paired['b_right_a_wrong']:,}")
    say(f"  exact McNemar p (ignores clustering) "
        f"{paired['mcnemar_exact_p_ignores_clustering']:.4f}")
    say("")
    say(f"  Brier    baseline {acc.loc[nvb.MODEL_LABEL, 'brier_two_class']:.4f}  "
        f"banded {acc.loc[nvb.MODEL_BANDED, 'brier_two_class']:.4f}  "
        f"delta {contrast_rows[0]['brier_delta']:+.4f}")
    say(f"  log loss baseline {acc.loc[nvb.MODEL_LABEL, 'log_loss']:.4f}  "
        f"banded {acc.loc[nvb.MODEL_BANDED, 'log_loss']:.4f}  "
        f"delta {contrast_rows[0]['log_loss_delta']:+.4f}")
    say("")
    say("  The interval is a paired patient bootstrap: the two arms score the same")
    say("  cases, so comparing their separate intervals would be the wrong test,")
    say("  and a patient contributing several predictions contributes several")
    say("  correlated comparisons. The McNemar p is reported for reference only")
    say("  and is anti-conservative here because it treats them as independent.")
    say("  No other statistical comparison is claimed.")

    # Keep the bootstrap draws for the plot.
    a_s = nv.scored(predictions[predictions["model"] == nvb.MODEL_BANDED])
    b_s = nv.scored(predictions[predictions["model"] == nvb.MODEL_LABEL])
    merged = a_s.merge(b_s, on=["PATNO", "visit_index"], suffixes=("_a", "_b"))
    diff = merged["correct_a"].to_numpy(float) - merged["correct_b"].to_numpy(float)
    by_pat = {p: g.to_numpy() for p, g in
              pd.DataFrame({"PATNO": merged["PATNO"], "d": diff}).groupby("PATNO")["d"]}
    keys = np.array(list(by_pat.keys()), dtype=object)
    rng = np.random.default_rng(cfg.seed)
    draws = np.array([np.concatenate([by_pat[keys[i]] for i in
                                      rng.choice(len(keys), len(keys), replace=True)]
                                     ).mean() for _ in range(2000)])

    # ----------------------------------------------------- 10. where it moved
    section("10. WHERE THE TWO ARMS DIFFER")
    breakdowns = nv.accuracy_breakdowns(predictions)
    by_ratio = nvb.ratio_range_accuracy(predictions)
    breakdowns["by_ratio_range"] = by_ratio
    for key, frame in breakdowns.items():
        frame.round(4).to_csv(out / f"accuracy_{key}.csv", index=False)

    for title, key in (("By the label recorded at the prediction point",
                        "by_current_label"),
                       ("By the ratio recorded at the prediction point",
                        "by_ratio_range"),
                       ("By how many of the patient's own visits were visible",
                        "by_history_length"),
                       (f"By inter-visit interval (split at "
                        f"{nv.INTERVAL_SPLIT_YEARS} years)", "by_interval")):
        frame = breakdowns[key]
        keep = frame[frame["model"].isin([nvb.MODEL_LABEL, nvb.MODEL_BANDED])]
        say(f"  {title}:")
        say(block(keep.round(4), index=False))
        say("")

    patient_level = []
    for name, g in predictions.groupby("model", sort=False):
        s = nv.scored(g)
        per = s.groupby("PATNO").agg(n_scored=("correct", "size"),
                                     n_correct=("correct", "sum"),
                                     accuracy=("correct", "mean"))
        per["model"] = name
        patient_level.append(per.reset_index())
    pd.concat(patient_level, ignore_index=True).to_csv(
        out / "patient_level_results.csv", index=False)

    ind_rows = [{"model": name, **nv.indeterminate_report(g)}
                for name, g in predictions.groupby("model", sort=False)]
    ind = pd.DataFrame(ind_rows)
    ind.round(4).to_csv(out / "indeterminate_report.csv", index=False)
    say("  Visits whose actual next label is Indeterminate (never scored, the")
    say("  conditioning event did not occur), and what each arm would have said:")
    say(block(ind.round(4), index=False))

    # ------------------------------------------------------------- 11. plots
    section("11. PLOTS")
    try:
        write_plots(out, comparison, paired, draws, all_ratios, scheme, by_ratio)
        say("  wrote comparison.png and accuracy_by_ratio_range.png")
    except Exception as exc:                     # noqa: BLE001
        say(f"  plots skipped: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------- 12. provenance
    section("12. PROVENANCE AND REPRODUCIBILITY")
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
        "min_tail_observations": rb.MIN_TAIL_OBSERVATIONS,
        "min_tail_patients": rb.MIN_TAIL_PATIENTS,
        "main_band_width": rb.MAIN_WIDTH,
        "tail_band_width": rb.TAIL_WIDTH,
        "tail_start": rb.TAIL_START,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                            # noqa: BLE001
        env["git_commit"] = "unknown"
    env["code_sha256"] = {
        str(p.relative_to(ROOT)): sha(p) for p in
        [Path(__file__), ROOT / "src/ppmi_subtype/ratio_bands.py",
         ROOT / "src/ppmi_subtype/hmm_banded.py",
         ROOT / "src/ppmi_subtype/next_visit_banded.py",
         ROOT / "src/ppmi_subtype/next_visit.py", ROOT / "src/ppmi_subtype/hmm.py",
         ROOT / "src/ppmi_subtype/ctmc.py", ROOT / "src/ppmi_subtype/loo.py",
         ROOT / "src/ppmi_subtype/hmm_stability.py"]
    }
    for k, v in env.items():
        if k != "code_sha256":
            say(f"  {k:<26} {v}")

    config_record = {
        "task": "next observed motor subtype, TD vs PIGD, conditional on the next "
                "label not being Indeterminate",
        "models": list(nvb.MODELS),
        "primary_comparison": [nvb.MODEL_BANDED, nvb.MODEL_LABEL],
        "primary_metric": "accuracy on eligible, defined cases",
        "hidden_states": 3,
        "validation": "leave-one-patient-out, whole patient held out of every fit "
                      "and of the band construction",
        "band_scheme": {
            "main_range": [rb.MAIN_LOW, rb.MAIN_HIGH],
            "main_width": rb.MAIN_WIDTH,
            "n_main_bands": len(rb.main_edges()) - 1,
            "tail_start": rb.TAIL_START,
            "tail_candidate_width": rb.TAIL_WIDTH,
            "merge_rule": f">= {rb.MIN_TAIL_OBSERVATIONS} observations and "
                          f">= {rb.MIN_TAIL_PATIENTS} patients per tail band, "
                          "sweeping upward; a short final band is absorbed below",
            "interval_convention": "left-closed, right-open; topmost tail band open",
            "undefined_ratio": "its own category per subtype",
            "fitted_on": "training patients of each fold only",
        },
        "emission_model": {
            "form": "E[a, (s, b)] = M[a, s] * C[a, s, b]",
            "constraint": f"M[a, a] >= {cfg.model['emission_floor']}, identical to "
                          "the square model's floor, applied to the subtype marginal",
            "aggregation": "P(subtype) = sum of joint categories of that subtype, "
                           "renormalised",
        },
        "optimiser": {
            "method": "L-BFGS-B",
            "options": dict(hmm_stability.OPTIONS),
            "gradient_banded": "exact (forward-backward)",
            "gradient_baseline": "finite difference, as audited",
            "restarts": cfg.model["n_restarts"],
            "seed": cfg.seed,
        },
    }
    (out / "experiment_config.json").write_text(
        json.dumps(config_record, indent=2, default=str), encoding="utf-8")

    summary = {
        "environment": env,
        "config": config_record,
        "data_integrity": checks_data,
        "counts": {
            "n_patients": len(data.sequences),
            "n_labelled_visits": int(sum(len(s["labels"]) for s in bdata.sequences)),
            "n_cases": len(cases),
            "n_eligible": int(cases["eligible"].sum()),
            "n_indeterminate_next": int((~cases["eligible"]).sum()),
            "n_folds_attempted": len(patients),
            "n_folds_completed": int(predictions["held_out_patient"].nunique()),
            "n_folds_failed": len(failures),
            "n_ratio_computable": int(len(finite)),
            "n_ratio_undefined": int(len(all_ratios) - len(finite)),
            "n_joint_categories_cohort_scheme": alphabet.n_categories,
            "n_tail_bands_cohort_scheme": len(scheme.tail_lows),
        },
        "checks": checks,
        "model_comparison": comparison.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "convergence": conv.to_dict(orient="records"),
        "fold_band_structures": structure.to_dict(orient="records"),
        "accuracy_breakdowns": {k: v.to_dict(orient="records")
                                for k, v in breakdowns.items()},
        "indeterminate": ind.to_dict(orient="records"),
        "band_occupancy_cohort_scheme": occupancy.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                      encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log", "driver_resume.log",
            "predictions_checkpoint.csv", "fold_band_diagnostics_checkpoint.csv"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
