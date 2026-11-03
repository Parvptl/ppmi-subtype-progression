"""E-1: resolve the lag-1 residual-autocorrelation contradiction.

THE CONTRADICTION
-----------------
`outputs/exploration/report.txt` draws two incompatible conclusions from one
number. Section 5 says

    "More positive than the null, which is the signature of accumulating
     process noise and favours the state-space form the proposal mentions."

and its own closing section says

    "4. Lag-1 residual correlation is -0.219 against a null of -0.332 ... so
        independent deviations are a defensible assumption and the growth form
        is adequate. Nothing here argues for the state-space alternative."

WHAT THE INSPECTION ESTABLISHED, BEFORE ANY RECOMPUTATION
---------------------------------------------------------
The phrase "Nothing here argues for the state-space alternative" **does not
exist anywhere in the committed source tree**. `scripts/06_ratio_exploration.py`
has been committed exactly once (`4e5bdd2`) and in that version both the
section-5 text and the closing item 4 are chosen by `if/elif/else` on the same
comparison of `r` against the null interval `[lo, hi]`. The two branches agree
by construction: when `r > hi` both print the state-space reading.

The commit message of `4e5bdd2` says so explicitly:

    "A first draft of this script asserted the opposite in its summary, having
     been written before the null was computed. It contradicted the analysis
     directly above it and has been corrected."

So the contradiction is **not a live code defect**. It is a stale artefact:
`outputs/exploration/report.txt` was written by the pre-fix draft, `outputs/`
was later made read-only, and the file was never regenerated. The README
carries the corrected reading.

WHAT THIS SCRIPT DOES
---------------------
1.  Recomputes the statistic and the simulated null with the committed code
    path and the committed seed, and checks both against the stale report's
    numbers.
2.  Strengthens the inference the stale report could not support: 2,000 null
    draws instead of 200, a one-sided Monte Carlo p-value, and a
    **patient-clustered bootstrap on the observed statistic** so the
    comparison carries uncertainty on both sides rather than only on the null.
3.  Re-runs `scripts/06_ratio_exploration.py` unmodified, with `paths.outputs`
    redirected into this run directory, and diffs the regenerated report
    against the stale one line by line -- demonstrating that the committed
    code is self-consistent and identifying exactly which lines are stale.
4.  Collects the independent evidence already in the repository that bears on
    the same question (the GPR kernel results), because a single
    autocorrelation statistic is thin ground for a modelling decision.

Nothing under `outputs/` is read for anything but input, and nothing is
written there.

    python scripts/76_lag1_residual_resolution.py --out <dir>
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402

STALE_REPORT = "outputs/exploration/report.txt"
STALE_PHRASE = "Nothing here argues for the state-space alternative"

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


def norm(text: str) -> str:
    """Collapse all whitespace, so a phrase wrapped across lines still matches.

    The stale sentence is split over two lines in the saved report, which an
    exact substring search silently misses.
    """
    return " ".join(text.split())


# ---------------------------------------------------------------------------
#  The statistic, reproducing scripts/06 exactly
# ---------------------------------------------------------------------------
def eligible_patients(finite: pd.DataFrame) -> list:
    """Patients script 06 admits to the autocorrelation statistic.

    Its rule, verbatim: at least 4 finite-ratio visits and at least one year
    of span. Applied per patient over the finite-ratio visits only.
    """
    keep = []
    for patno, g in finite.groupby("PATNO", sort=True):
        if len(g) < 4:
            continue
        t = g["years_from_baseline"].to_numpy()
        if t.max() - t.min() < 1.0:
            continue
        keep.append(patno)
    return keep


def residual_pairs(finite: pd.DataFrame, patients) -> np.ndarray:
    """Consecutive OLS-residual pairs, one patient's own straight line each."""
    pairs = []
    for patno in patients:
        g = finite[finite["PATNO"] == patno]
        t = g["years_from_baseline"].to_numpy()
        y = g["ratio"].to_numpy()
        resid = y - np.polyval(np.polyfit(t, y, 1), t)
        pairs.extend(zip(resid[:-1], resid[1:]))
    return np.asarray(pairs)


def lag1(arr: np.ndarray) -> float:
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def within_patient_sd(finite: pd.DataFrame) -> float:
    """sqrt(MS_within), exactly as scripts/06 section 3 computes it."""
    per_n = finite.groupby("PATNO")["ratio"].size()
    multi = finite[finite["PATNO"].isin(per_n[per_n >= 2].index)]
    means = multi.groupby("PATNO")["ratio"].mean()
    ss_within = float(((multi["ratio"] - multi["PATNO"].map(means)) ** 2).sum())
    df_within = len(multi) - len(means)
    return float(np.sqrt(ss_within / df_within))


def simulate_null(finite: pd.DataFrame, patients, sigma: float,
                  n_draws: int, seed: int) -> np.ndarray:
    """iid noise on the same visit pattern, the committed generative recipe.

    scripts/06 draws y = 2.0 - 0.09 t + N(0, sigma) per visit, refits each
    patient's line, and correlates consecutive residuals. The trend is
    irrelevant to the statistic -- detrending removes it -- but it is kept
    here so the code path is the committed one.
    """
    rng = np.random.default_rng(seed)
    times = [finite.loc[finite["PATNO"] == p, "years_from_baseline"].to_numpy()
             for p in patients]
    out = np.empty(n_draws)
    for d in range(n_draws):
        sim = []
        for t in times:
            y = 2.0 - 0.09 * t + rng.normal(0.0, sigma, len(t))
            e = y - np.polyval(np.polyfit(t, y, 1), t)
            sim.extend(zip(e[:-1], e[1:]))
        out[d] = lag1(np.asarray(sim))
    return out


def cluster_bootstrap_observed(finite: pd.DataFrame, patients, n_boot: int,
                               seed: int) -> np.ndarray:
    """Patient-clustered bootstrap of the OBSERVED lag-1 correlation.

    The stale report puts an interval on the null and a point estimate on the
    observation. A patient resample puts one on the observation too, which is
    what the comparison actually needs: 20 of these patients contribute six or
    more residual pairs.
    """
    rng = np.random.default_rng(seed)
    per_patient = {p: None for p in patients}
    for p in patients:
        g = finite[finite["PATNO"] == p]
        t = g["years_from_baseline"].to_numpy()
        y = g["ratio"].to_numpy()
        r = y - np.polyval(np.polyfit(t, y, 1), t)
        per_patient[p] = np.column_stack([r[:-1], r[1:]])

    blocks = [per_patient[p] for p in patients]
    n = len(blocks)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        arr = np.vstack([blocks[i] for i in pick])
        out[b] = lag1(arr)
    return out


# ---------------------------------------------------------------------------
#  Re-running scripts/06 with outputs redirected
# ---------------------------------------------------------------------------
def rerun_script_06(out: Path) -> tuple[Path, Path]:
    """Run the committed script unmodified into a shim outputs tree.

    `paths.outputs` is overridden to a directory inside this run; the frozen
    panel is symlinked in so the script finds its input where it expects it.
    Returns (shim config path, regenerated report path).
    """
    shim_out = out / "rerun_outputs"
    shim_out.mkdir(parents=True, exist_ok=True)

    panel_link = shim_out / "visit_panel.csv"
    if not panel_link.exists():
        panel_link.symlink_to(ROOT / "outputs" / "visit_panel.csv")

    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"]["outputs"] = str(shim_out.relative_to(ROOT))
    shim_cfg = out / "config_rerun.yaml"
    shim_cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    env = {**_os_environ(), "PPMI_CONFIG": str(shim_cfg)}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "06_ratio_exploration.py")],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    (out / "rerun_stdout.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"scripts/06 exited {proc.returncode}; see rerun_stdout.log")
    return shim_cfg, shim_out / "exploration" / "report.txt"


def _os_environ() -> dict:
    import os
    return dict(os.environ)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=2000,
                    help="null draws (scripts/06 uses 200)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = load_config()
    checks: dict[str, bool] = {}

    section("E-1  LAG-1 RESIDUAL AUTOCORRELATION: RESOLVING THE CONTRADICTION")

    # ---------------------------------------------------------------- inputs
    panel_path = ROOT / "outputs" / "visit_panel.csv"
    panel = pd.read_csv(panel_path, low_memory=False)
    md5 = hashlib.md5(panel_path.read_bytes()).hexdigest()
    say(f"  panel                {panel_path.relative_to(ROOT)}")
    say(f"  panel md5            {md5}")
    checks["frozen panel md5 matches the project's recorded value"] = (
        md5 == "fb038ddbdc19ab3368e13940f398e7ba")

    labelled = panel[panel["subtype"].notna()]
    finite = labelled[labelled["ratio"].notna() & np.isfinite(labelled["ratio"])]
    finite = finite.sort_values(["PATNO", "years_from_baseline"])
    say(f"  labelled visits      {len(labelled):,}")
    say(f"  finite-ratio visits  {len(finite):,}")

    # ------------------------------------------------- the stale report text
    section("1. THE STALE ARTEFACT, LOCATED")
    stale = (ROOT / STALE_REPORT).read_text(encoding="utf-8")
    src06 = (ROOT / "scripts" / "06_ratio_exploration.py").read_text(encoding="utf-8")

    # The sentence is wrapped across two lines in the saved report, so the
    # comparison is made on whitespace-collapsed text.
    in_report = norm(STALE_PHRASE) in norm(stale)
    in_source = norm(STALE_PHRASE) in norm(src06)
    say(f"  phrase under audit:  {STALE_PHRASE!r}")
    say(f"  present in {STALE_REPORT}: {in_report}")
    say(f"  present in scripts/06_ratio_exploration.py: {in_source}")
    checks["the contradictory sentence is present in the stale saved report"] = in_report
    checks["the contradictory sentence is absent from the committed script"] = not in_source

    commits = subprocess.run(
        ["git", "log", "--format=%h", "--", "scripts/06_ratio_exploration.py"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    say(f"  commits touching scripts/06: {len(commits)} ({', '.join(commits) or 'none'})")
    checks["scripts/06 has been committed exactly once"] = len(commits) == 1

    # Two searches. "Nothing here" alone is a common phrase and matches three
    # unrelated committed files, so the distinctive clause is used instead.
    distinctive = "argues for the state-space alternative"
    g = subprocess.run(["git", "grep", "-l", distinctive, "--"] + ["."],
                       cwd=ROOT, capture_output=True, text=True)
    tree_hits = g.stdout.strip()
    say(f"  git grep {distinctive!r} across the working tree: "
        f"{tree_hits or 'no match'}")
    checks["the distinctive clause appears in no file in the working tree"] = not tree_hits

    # And every commit's own version of script 06, whitespace-collapsed, so a
    # line-wrapped occurrence cannot hide from a line-oriented search.
    revs = subprocess.run(["git", "rev-list", "--all"], cwd=ROOT,
                          capture_output=True, text=True).stdout.split()
    hist_hits = []
    for rev in revs:
        blob = subprocess.run(
            ["git", "show", f"{rev}:scripts/06_ratio_exploration.py"],
            cwd=ROOT, capture_output=True, text=True)
        if blob.returncode == 0 and norm(STALE_PHRASE) in norm(blob.stdout):
            hist_hits.append(rev)
    say(f"  whitespace-insensitive search of scripts/06 across all "
        f"{len(revs)} commits: {hist_hits or 'no match'}")
    checks["the sentence appears in no committed version of scripts/06"] = not hist_hits

    say("\n  So the contradiction is a STALE OUTPUT, not a live code defect:")
    say("  the committed script cannot produce that sentence.")

    # ----------------------------------------------- reproduce the statistic
    section("2. REPRODUCING THE STATISTIC WITH THE COMMITTED CODE PATH")
    pats = eligible_patients(finite)
    arr = residual_pairs(finite, pats)
    r_obs = lag1(arr)
    sigma = within_patient_sd(finite)
    say(f"  eligible patients (>=4 finite visits, >=1y span)  {len(pats):,}")
    say(f"  consecutive residual pairs                        {len(arr):,}")
    say(f"  observed lag-1 correlation                         {r_obs:+.4f}")
    say(f"  sigma_hat = sqrt(MS_within)                        {sigma:.4f}")

    checks["residual-pair count reproduces the stale report's 2,050"] = len(arr) == 2050
    checks["observed lag-1 correlation reproduces the stale report's -0.219"] = (
        round(r_obs, 3) == -0.219)

    null_committed = simulate_null(finite, pats, sigma, 200, cfg.seed)
    lo_c, hi_c = np.quantile(null_committed, [0.025, 0.975])
    say(f"\n  null, committed recipe (200 draws, seed {cfg.seed}):")
    say(f"    mean {null_committed.mean():+.4f}  95% [{lo_c:+.4f}, {hi_c:+.4f}]")
    checks["null mean reproduces the stale report's -0.332"] = (
        round(float(null_committed.mean()), 3) == -0.332)
    checks["null interval reproduces the stale report's [-0.365, -0.294]"] = (
        round(float(lo_c), 3) == -0.365 and round(float(hi_c), 3) == -0.294)

    say("\n  Every published number reproduces. The arithmetic was never in doubt;")
    say("  only the sentence drawn from it was.")

    # ------------------------------------------------- strengthen the result
    section("3. A STRONGER TEST THAN THE STALE REPORT COULD SUPPORT")
    say(f"  {args.n_null:,} null draws instead of 200, a one-sided Monte Carlo")
    say(f"  p-value, and a {args.n_boot:,}-draw patient-clustered bootstrap on the")
    say("  OBSERVED statistic -- the stale report put an interval on the null only.\n")

    null = simulate_null(finite, pats, sigma, args.n_null, cfg.seed)
    lo, hi = np.quantile(null, [0.025, 0.975])
    # One-sided: is the observation MORE POSITIVE than iid noise would give?
    p_one = float((null >= r_obs).sum() + 1) / (args.n_null + 1)

    boot = cluster_bootstrap_observed(finite, pats, args.n_boot, cfg.seed)
    b_lo, b_hi = np.quantile(boot, [0.025, 0.975])
    # Difference observed - null, resampling both independently.
    rng = np.random.default_rng(cfg.seed + 1)
    diff = (rng.choice(boot, args.n_boot, replace=True)
            - rng.choice(null, args.n_boot, replace=True))
    d_lo, d_hi = np.quantile(diff, [0.025, 0.975])

    tbl = pd.DataFrame([
        {"quantity": "observed lag-1 correlation", "estimate": r_obs,
         "ci_low": b_lo, "ci_high": b_hi, "interval": "patient bootstrap"},
        {"quantity": "iid null, same visit pattern", "estimate": float(null.mean()),
         "ci_low": lo, "ci_high": hi, "interval": "Monte Carlo"},
        {"quantity": "observed minus null", "estimate": r_obs - float(null.mean()),
         "ci_low": d_lo, "ci_high": d_hi, "interval": "combined resample"},
    ])
    tbl.to_csv(out / "lag1_statistic.csv", index=False)
    say(frame(tbl.round(4)))
    say(f"\n  one-sided Monte Carlo p (null >= observed): {p_one:.2e}")

    outside = not (lo <= r_obs <= hi)
    more_positive = r_obs > hi
    say(f"\n  observed outside the null interval:  {outside}")
    say(f"  observed MORE POSITIVE than the null: {more_positive}")
    checks["observed statistic lies outside the null interval"] = outside
    checks["observed statistic is more positive than the null"] = more_positive
    checks["the observed-minus-null interval excludes zero"] = d_lo > 0
    checks["conclusion is unchanged at 10x the null draws"] = more_positive

    np.save(out / "null_draws.npy", null)
    np.save(out / "bootstrap_draws.npy", boot)

    # -------------------------------------------------------------- the diff
    section("4. RE-RUNNING THE COMMITTED SCRIPT, AND DIFFING AGAINST THE STALE REPORT")
    say("  scripts/06 is run unmodified with paths.outputs redirected into this")
    say("  run directory. outputs/ is read-only and is not touched.\n")
    shim_cfg, regen_path = rerun_script_06(out)
    regen = regen_path.read_text(encoding="utf-8")
    say(f"  regenerated report: {regen_path.relative_to(out)}")
    say(f"  shim config:        {shim_cfg.relative_to(out)}")

    checks["the regenerated report does not contain the contradictory sentence"] = (
        norm(STALE_PHRASE) not in norm(regen))
    # Self-consistency: both passages must now carry the same reading.
    s5 = "More positive than the null, which is the signature of accumulating"
    i4 = "MORE POSITIVE than the null and outside its interval"
    has_s5, has_i4 = norm(s5) in norm(regen), norm(i4) in norm(regen)
    say(f"\n  section 5 carries the state-space reading:        {has_s5}")
    say(f"  closing item 4 carries the state-space reading:  {has_i4}")
    checks["regenerated section 5 carries the state-space reading"] = has_s5
    checks["regenerated closing item 4 carries the same reading"] = has_i4
    checks["the regenerated report is internally self-consistent"] = has_s5 and has_i4
    # And the stale report must fail exactly that self-consistency test.
    checks["the stale report fails the same self-consistency test"] = (
        norm(s5) in norm(stale) and norm(i4) not in norm(stale))

    diff_lines = list(difflib.unified_diff(
        stale.splitlines(), regen.splitlines(),
        fromfile=STALE_REPORT, tofile="regenerated (committed code)", lineterm="", n=2))
    (out / "report_diff.txt").write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
    changed = [l for l in diff_lines if l.startswith(("+", "-"))
               and not l.startswith(("+++", "---"))]
    say(f"\n  differing lines: {len(changed)}  (full diff in report_diff.txt)")
    say("\n  The substantive change, stale -> regenerated:")
    for l in diff_lines:
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---")):
            if "state-space" in l or "growth form" in l or "independent deviations" in l:
                say("    " + l)

    # -------------------------------------------- independent corroboration
    section("5. INDEPENDENT EVIDENCE IN THE REPOSITORY ON THE SAME QUESTION")
    say("  One autocorrelation statistic is thin ground for a modelling choice.")
    say("  Two completed experiments bear on it independently, and both point the")
    say("  same way. Neither was run for this purpose.\n")
    say("  (a) GPR kernel comparison (docs/gpr_audit/FINAL_RESULTS.md section 3.1).")
    say("      K1 = constant + RBF beats K0 = constant + linear on held-out MAE in")
    say("      all four cohorts under both aggregations: unfiltered -0.0333")
    say("      [-0.0467, -0.0205] pooled, -0.0192 [-0.0279, -0.0106] per patient.")
    say("      An RBF term with a ~3.7-year lengthscale IS temporally persistent")
    say("      deviation from the trend. That is the same structure a state-space")
    say("      form posits, reached by a different route.")
    say("")
    say("  (b) The linear-kernel variance pins at its lower bound (1e-6) in")
    say("      1642/1642 unfiltered fits, and in 821/821 folds of the zero-mean")
    say("      arm as well (same document, section 4.1 and its correction).")
    say("      So the data rejects a patient-specific random SLOPE while retaining")
    say("      a persistent-deviation term -- which is what 'process noise")
    say("      accumulates' means and is not what 'patients differ in slope' means.")
    say("")
    say("  (c) The growth model's own identifiability note (src/.../hbayes.py):")
    say("      sigma_obs and sigma_proc enter the likelihood only through their")
    say("      sum, so the growth form cannot separate them. The state-space form")
    say("      can, because accumulating process noise leaves a distinguishable")
    say("      temporal signature -- the very signature measured above.")

    # ------------------------------------------------------------- verdict
    section("6. VERDICT")
    say("  RESOLVED, and the resolution is not a judgement call.")
    say("")
    say("  * The committed code is self-consistent. The contradiction exists only")
    say("    in outputs/exploration/report.txt, written by an uncommitted earlier")
    say("    draft and never regenerated because outputs/ became read-only.")
    say("  * Section 5 of that report is CORRECT. The closing item 4 is STALE and")
    say("    must not be quoted.")
    say(f"  * The observed correlation ({r_obs:+.3f}) is more positive than iid noise")
    say(f"    on the same visit pattern would give ({null.mean():+.3f}), the difference")
    say(f"    interval excludes zero ([{d_lo:+.4f}, {d_hi:+.4f}]), and the one-sided")
    say(f"    Monte Carlo p-value is {p_one:.2e}.")
    say("  * Deviations from a patient's own trend PERSIST between visits. That")
    say("    favours the state-space form over the growth form, corroborated")
    say("    independently by the GPR kernel results.")
    say("")
    say("  FOR THE MANUSCRIPT. The defensible sentence is: residual deviations from")
    say("  a patient's own linear trend are more persistent than independent")
    say("  measurement error would produce, which favours a state-space over a")
    say("  growth specification -- and is a second, independent reason to prefer it,")
    say("  the first being that only the state-space form identifies the")
    say("  observation and process variances separately.")
    say("")
    say("  WHAT IT DOES NOT LICENSE. The project never fitted the state-space form")
    say("  to the ratio, so this is a statement about model specification and not")
    say("  a claim that a state-space model would predict better. The one")
    say("  state-space-like model that WAS fitted -- the rich clinical profile's")
    say("  local linear trend -- found the rate component worth +0.0040 of +0.0733")
    say("  (about 5%), so the expected gain on this evidence is small.")

    # ------------------------------------------------------------- artefacts
    section("7. CHECKS")
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
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "node": platform.node(), "seed": cfg.seed,
            "n_null_draws": args.n_null, "n_bootstrap": args.n_boot,
            "elapsed_seconds": round(time.time() - started, 1),
            "git_commit": commit, "panel_md5": md5,
            "models_fitted": 0, "analysis_only": True,
        },
        "statistic": {
            "n_patients": len(pats), "n_residual_pairs": int(len(arr)),
            "observed_lag1": r_obs,
            "observed_ci": [float(b_lo), float(b_hi)],
            "null_mean": float(null.mean()),
            "null_ci": [float(lo), float(hi)],
            "difference": r_obs - float(null.mean()),
            "difference_ci": [float(d_lo), float(d_hi)],
            "one_sided_mc_p": p_one,
            "sigma_hat": sigma,
        },
        "verdict": {
            "contradiction_is_a_stale_output_not_a_code_defect": True,
            "correct_reading": "state-space form favoured",
            "stale_passage": "outputs/exploration/report.txt closing item 4",
            "correct_passage": "outputs/exploration/report.txt section 5",
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "code_sha256": {"scripts/76_lag1_residual_resolution.py": sha(__file__),
                        "scripts/06_ratio_exploration.py":
                            sha(ROOT / "scripts" / "06_ratio_exploration.py")},
    }, indent=1), encoding="utf-8")

    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "driver.log").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*")
                   if p.is_file() and not p.is_symlink() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")

    print(f"\nWrote {len(files)} files to {out}")
    return 0 if n_ok == len(cf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
