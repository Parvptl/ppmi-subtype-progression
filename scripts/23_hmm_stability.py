"""Stage 4: systematic HMM stability study, with selection kept off the test set.

Configurations (all predict the next recorded 3-class label):

    C3_free      3 latent states, no emission floor
    C3_f50       3 latent states, floor 0.50  (the audited default)
    C3_f60       3 latent states, floor 0.60
    C3_f70       3 latent states, floor 0.70
    C4_free      4 latent states (rectangular emission, no floor)
    C5_free      5 latent states (as the Step 2 five-state LOO)
    C3_f50_audited   `hmm.fit_hmm(..., n_restarts=4, seed=42)` exactly as the
                     original LOO ran it; the reference everything is paired to

Stages
------
population   EXPLORATORY. All 934 patients, `--pop-starts` dispersed starts per
             configuration. Describes the likelihood surface: how many starts
             reach the best optimum, distinct optima, parameter spread at the
             optimum, termination reasons. Not a held-out result.

cv           CONFIRMATORY. Outer K-fold patient-level CV (`--outer 10`). Inside
             each outer fold every configuration is fitted on the training
             patients from `--starts` dispersed starts; the best *training*
             likelihood is kept (that is the MLE, not selection on test). Each
             start's held-out log-likelihood is ALSO recorded, so variation in
             held-out performance across initialisations is measured.

             Configuration selection is done twice, both without the outer test
             patients: (a) inner `--inner`-fold CV log-likelihood on the
             training patients; (b) training BIC. The selected configuration's
             outer-fold predictions are the unbiased estimate of "choose a
             configuration, then predict". The per-configuration outer results
             are reported as a sensitivity table, never as the selected result.

Every task writes its own JSON/CSV under the run directory and is skipped on
resume, so a killed job loses only the tasks in flight.

    python scripts/23_hmm_stability.py --out <run_dir> --stage population|cv
        [--jobs 60] [--pop-starts 40] [--starts 10] [--outer 10] [--inner 5]
        [--configs C3_f50,C5_free,...] [--resume]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import ctmc, hmm, loo  # noqa: E402
from ppmi_subtype import hmm_stability as HS  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
NAMES = loo.STATE_NAMES

CONFIGS = {
    "C3_free": (3, None), "C3_f50": (3, 0.5), "C3_f60": (3, 0.6),
    "C3_f70": (3, 0.7), "C4_free": (4, None), "C5_free": (5, None),
    "C3_f50_audited": (3, 0.5),
}
SELECTABLE = ["C3_free", "C3_f50", "C3_f60", "C3_f70", "C4_free", "C5_free"]

_S: dict = {}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fit_from_theta(theta, n_hidden, floor, states, n_seq, n_obs):
    Q, E, pi = hmm._unpack(theta, n_hidden, len(states))
    return hmm.HMMFit(Q=Q, E=E, pi=pi, states=list(states), state_names=NAMES,
                      loglik=float("nan"), n_sequences=n_seq, n_observations=n_obs,
                      n_params=hmm._n_params(n_hidden, len(states)), converged=True,
                      emission_floor=floor, n_hidden=n_hidden,
                      hidden_names=(NAMES if n_hidden == 3
                                    else [f"H{k + 1}" for k in range(n_hidden)]))


def heldout_loglik(fit, test_seqs):
    """Sum and count of log P(next label | history) over the test transitions.

    The first label of each sequence is not scored, matching the one-step-ahead
    evaluation in `loo` (3,361 transitions over 934 patients).
    """
    p = hmm.predictive_distributions(fit, test_seqs)
    idx = {s: k for k, s in enumerate(fit.states)}
    probs = p[[f"p_{n}" for n in NAMES]].to_numpy()
    j = p["to_state"].map(idx).to_numpy(int)
    return float(np.log(np.clip(probs[np.arange(len(j)), j], 1e-300, None)).sum()), len(j), p


def _fit_config(config, train, starts_n, seed):
    """Returns (best fit, records). The audited arm uses `hmm.fit_hmm` itself."""
    n_hidden, floor = CONFIGS[config]
    states = _S["states"]
    if config == "C3_f50_audited":
        f = hmm.fit_hmm(train, states, state_names=NAMES, seed=SEED, n_restarts=4,
                        emission_floor=0.5)
        rec = HS.StartRecord(start=0, loglik=f.loglik, success=f.converged,
                             message="hmm.fit_hmm (message not exposed)",
                             n_iter=-1, n_fev=-1, proj_grad=float("nan"),
                             theta=np.full(hmm._n_params(3, 3), np.nan))
        return f, [rec]
    starts = HS.dispersed_starts(n_hidden, len(states), floor, starts_n, seed)
    return HS.fit_multistart(train, states, n_hidden, floor, starts, state_names=NAMES,
                             maxfun=_S.get("maxfun"))


def _record_dicts(records, n_hidden, n_obs_labels, test_seqs=None, train_meta=None):
    out = []
    for r in records:
        d = {"start": r.start, "loglik": r.loglik, "success": r.success,
             "message": r.message, "n_iter": r.n_iter, "n_fev": r.n_fev,
             "proj_grad": r.proj_grad}
        if np.all(np.isfinite(r.theta)):
            Q, E, pi = HS.canonical_order(*hmm._unpack(r.theta, n_hidden, n_obs_labels))
            d.update({"Q": Q.tolist(), "E": E.tolist(), "pi": pi.tolist()})
            if test_seqs is not None:
                f = fit_from_theta(r.theta, n_hidden, None, _S["states"], *train_meta)
                ll, n, _ = heldout_loglik(f, test_seqs)
                d.update({"test_loglik": ll, "test_n": n})
        out.append(d)
    return out


def _atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
#  Tasks
# ---------------------------------------------------------------------------
def task_population(config, start_idx):
    """One dispersed start on the whole cohort (exploratory)."""
    out = _S["run_dir"] / "population" / config / f"start_{start_idx:03d}.json"
    if out.exists():
        return "skipped"
    n_hidden, floor = CONFIGS[config]
    seqs, states = _S["seqs"], _S["states"]
    starts = HS.dispersed_starts(n_hidden, len(states), floor, _S["pop_starts"], SEED)
    t0 = time.perf_counter()
    _, recs = HS.fit_multistart(seqs, states, n_hidden, floor, starts[[start_idx]],
                                maxfun=_S.get("maxfun"))
    d = _record_dicts(recs, n_hidden, len(states))[0]
    d.update({"start": start_idx, "config": config, "seconds": time.perf_counter() - t0,
              "n_params": hmm._n_params(n_hidden, len(states)),
              "n_observations": int(sum(len(s["labels"]) for s in seqs))})
    _atomic_json(d, out)
    return "done"


def task_cv(config, outer, inner):
    """Fit `config` on a training set; score the matching held-out patients.

    inner is None  -> train = outer-training patients, test = outer fold
    inner = j      -> train = outer-training minus inner fold j, test = inner fold j
    """
    tag = f"outer{outer}" + ("" if inner is None else f"_inner{inner}")
    out = _S["run_dir"] / "cv" / config / f"{tag}.json"
    if out.exists():
        return "skipped"
    seqs, of, inf_ = _S["seqs"], _S["outer_fold"], _S["inner_fold"]
    train_all = [s for s in seqs if of[s["PATNO"]] != outer]
    if inner is None:
        train, test = train_all, [s for s in seqs if of[s["PATNO"]] == outer]
    else:
        fold = inf_[outer]
        train = [s for s in train_all if fold[s["PATNO"]] != inner]
        test = [s for s in train_all if fold[s["PATNO"]] == inner]
    if {s["PATNO"] for s in train} & {s["PATNO"] for s in test}:
        raise RuntimeError(f"leakage in {config} {tag}")

    t0 = time.perf_counter()
    n_hidden, floor = CONFIGS[config]
    seed = SEED + 1000 * outer + (0 if inner is None else 100 * (inner + 1))
    fit, recs = _fit_config(config, train, _S["starts"], seed)
    meta = (len(train), int(sum(len(s["labels"]) for s in train)))
    ll, n, preds = heldout_loglik(fit, test)
    d = {"config": config, "outer": outer, "inner": inner,
         "n_train_patients": len(train), "n_test_patients": len(test),
         "train_loglik": fit.loglik, "n_params": fit.n_params,
         "n_train_obs": meta[1], "test_loglik": ll, "test_n": n,
         "best_Q": HS.canonical_order(fit.Q, fit.E, fit.pi)[0].tolist(),
         "best_E": HS.canonical_order(fit.Q, fit.E, fit.pi)[1].tolist(),
         "best_pi": HS.canonical_order(fit.Q, fit.E, fit.pi)[2].tolist(),
         "records": _record_dicts(recs, n_hidden, 3,
                                  test_seqs=test if inner is None else None,
                                  train_meta=meta),
         "seconds": time.perf_counter() - t0}
    if inner is None:
        # Training-fold label prevalence for the prior-corrected rule, and the
        # held-out predictions themselves.
        lab = np.concatenate([s["labels"][1:] for s in train])
        d["train_prior"] = [float((lab == k).mean()) for k in range(3)]
        preds = preds.rename(columns={"step": "visit_index"})
        preds["config"], preds["outer"] = config, outer
        p_out = out.with_suffix(".csv")
        p_out.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(p_out.with_suffix(".tmp"), index=False)
        os.replace(p_out.with_suffix(".tmp"), p_out)
    _atomic_json(d, out)
    return "done"


def task_baselines(outer):
    """Observed-label chains and persistence on the outer test fold."""
    out = _S["run_dir"] / "cv" / "_baselines" / f"outer{outer}.csv"
    if out.exists():
        return "skipped"
    data, of, states = _S["data"], _S["outer_fold"], _S["states"]
    train_pairs = data.pairs[data.pairs["PATNO"].map(of) != outer]
    test_pairs = data.pairs[data.pairs["PATNO"].map(of) == outer].reset_index(drop=True)
    models = {"Markov (discrete)": ctmc.fit_dtmc(train_pairs, states),
              "Marginal (null)": ctmc.fit_marginal(train_pairs, states),
              "CTMC (interval-aware)": ctmc.fit_ctmc(train_pairs, states,
                                                     state_names=NAMES, seed=SEED,
                                                     n_restarts=4)}
    idx = {s: k for k, s in enumerate(states)}
    frames = []
    pers = np.eye(3)[test_pairs["from_state"].map(idx).to_numpy(int)]
    for name, m in [*models.items(), ("Persistence (stay)", None)]:
        probs = pers if m is None else ctmc.predict_pairs(m, test_pairs, states)
        f = test_pairs[["PATNO", "from_state", "to_state", "interval_years"]].copy()
        f["visit_index"] = test_pairs.groupby("PATNO").cumcount()
        for k, n in enumerate(NAMES):
            f[f"p_{n}"] = probs[:, k]
        f["config"], f["outer"] = name, outer
        frames.append(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out.with_suffix(".tmp"), index=False)
    os.replace(out.with_suffix(".tmp"), out)
    return "done"


def _init(state):
    _S.update(state)


def _dispatch(kind, *a):
    return {"population": task_population, "cv": task_cv,
            "baselines": task_baselines}[kind](*a)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["population", "cv"], required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--pop-starts", type=int, default=40)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--maxfun", type=int, default=None,
                    help="L-BFGS-B function-evaluation cap (default: scipy's 15000, "
                         "as the audited fitter)")
    ap.add_argument("--limit-patients", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="allow a non-frozen panel (synthetic smoke tests only)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = Path(args.out)
    if (run_dir / args.stage).exists() and not args.resume:
        raise SystemExit(f"{run_dir / args.stage} exists; pass --resume to continue it")
    run_dir.mkdir(parents=True, exist_ok=True)
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    if panel_md5 != EXPECTED_PANEL_MD5 and not args.smoke:
        raise SystemExit(f"panel MD5 {panel_md5} != frozen {EXPECTED_PANEL_MD5}")

    L = cfg.subtype.labels
    states = [L["TD"], L["Indeterminate"], L["PIGD"]]
    panel = pd.read_csv(panel_path, low_memory=False)
    data = loo.prepare(panel, states)
    seqs = data.sequences
    if args.limit_patients:
        keep = {s["PATNO"] for s in seqs[:args.limit_patients]}
        seqs = [s for s in seqs if s["PATNO"] in keep]
        data = loo.LOOData(sequences=seqs, pairs=data.pairs[data.pairs.PATNO.isin(keep)],
                           states=data.states, meta=data.meta[data.meta.PATNO.isin(keep)])
    patients = [int(s["PATNO"]) for s in seqs]
    outer_fold = HS.patient_folds(patients, args.outer, SEED)
    inner_fold = {o: HS.patient_folds([p for p in patients if outer_fold[p] != o],
                                      args.inner, SEED + 1 + o)
                  for o in range(args.outer)}
    configs = args.configs.split(",")

    meta = {"stage": args.stage, "argv": sys.argv, "seed": SEED, "configs": configs,
            "config_defs": {c: CONFIGS[c] for c in configs},
            "state_codes": [int(x) for x in states],
            "pop_starts": args.pop_starts, "starts": args.starts,
            "smoke_test": bool(args.smoke), "limit_patients": args.limit_patients,
            "outer": args.outer, "inner": args.inner, "panel_md5": panel_md5,
            "n_patients": len(patients),
            "n_transitions": int(sum(len(s["gaps"]) for s in seqs)),
            "optimizer_options": HS.OPTIONS, "maxfun": args.maxfun,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (run_dir / f"run_config_{args.stage}.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({"PATNO": patients, "outer_fold": [outer_fold[p] for p in patients]}
                 ).to_csv(run_dir / "outer_folds.csv", index=False)

    state = {"run_dir": run_dir, "seqs": seqs, "states": states, "data": data,
             "outer_fold": outer_fold, "inner_fold": inner_fold,
             "pop_starts": args.pop_starts, "starts": args.starts, "maxfun": args.maxfun}
    heavy = {"C5_free": 0, "C4_free": 1}
    if args.stage == "population":
        tasks = [("population", c, i) for c in configs if c != "C3_f50_audited"
                 for i in range(args.pop_starts)]
    else:
        tasks = [("baselines", o) for o in range(args.outer)]
        for c in configs:
            for o in range(args.outer):
                tasks.append(("cv", c, o, None))
                if c in SELECTABLE:
                    tasks += [("cv", c, o, j) for j in range(args.inner)]
    tasks.sort(key=lambda t: heavy.get(t[1], 9) if len(t) > 2 else 9)
    print(f"{len(tasks)} tasks, {args.jobs} workers, {len(patients)} patients", flush=True)

    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(state,)) as pool:
        futs = {pool.submit(_dispatch, *t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception:                                   # noqa: BLE001
                failed += 1
                with open(run_dir / f"failed_{args.stage}.jsonl", "a") as f:
                    f.write(json.dumps({"task": list(map(str, futs[fut])),
                                        "error": traceback.format_exc()}) + "\n")
            done += 1
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el / 60:.1f} min  failed {failed}",
                      flush=True)
    print(f"finished {args.stage}: {done} tasks, {failed} failed, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
