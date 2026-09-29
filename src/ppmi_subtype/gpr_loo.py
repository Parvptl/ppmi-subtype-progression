"""Patient-level leave-one-out validation of the trajectory models.

The design is the one `loo.py` already uses for the Markov models, so the two
sets of results mean the same thing by construction. For each patient in turn:

* every model is refitted from scratch on the *other* patients;
* the held-out patient's visit k is predicted from visits 0..k-1 only;
* the target and everything after it are never available, to the fit or to the
  prediction.

What is being predicted is different, though, and the difference matters when
reading any comparison. The Markov models predict the next categorical subtype.
These models predict the next *ratio*, as a distribution, and the subtype
probabilities are derived from it afterwards. A model can be good at one and bad
at the other.

Models scored here
------------------
    K0..K3        Gaussian processes, shared hyperparameters per fold
    hbayes        the existing hierarchical growth model, unmodified
    persistence   next ratio = last observed ratio
    patient_mean  next ratio = mean of the patient's own history
    population    next ratio = beta0 + beta1 t, pooled OLS on training patients
    patient_ols   least-squares line through the patient's history, extrapolated

The three fitted baselines get a predictive standard deviation estimated from
*training-fold* residuals, never from the held-out patient. Without one they
could only be scored on MAE and RMSE, and the interval and probability metrics
would have nothing to compare the models against.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import gpr, hbayes

STATE_NAMES = ["TD", "Indeterminate", "PIGD"]
PROB_COLS = [f"p_{s}" for s in STATE_NAMES]

BASELINES = ["persistence", "patient_mean", "population", "patient_ols"]


class LeakageError(RuntimeError):
    """Raised when a held-out patient is found in a training set."""


# ---------------------------------------------------------------------------
#  Fold data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoldData:
    """Per-patient arrays, small enough to hand to a worker process.

    Deliberately not the cohort DataFrame: a fold needs times, ratios and the
    observed subtype, and shipping the full panel to every worker would cost
    more than the fit.
    """

    times: dict          # PATNO -> (n,) float
    ratios: dict         # PATNO -> (n,) float
    subtypes: dict       # PATNO -> (n,) str

    @property
    def patients(self) -> list:
        return sorted(self.times)

    def sequences(self, exclude=None) -> list:
        """Training sequences, with `exclude` removed.

        The single place a training set is assembled. `run_fold` asserts the
        held-out patient is absent afterwards, so the exclusion is checked
        rather than trusted.
        """
        return [
            {"t": self.times[p], "y": self.ratios[p]}
            for p in self.patients
            if p != exclude
        ]


def prepare(cohort) -> FoldData:
    """Extract the fold arrays from a :class:`~ppmi_subtype.trajectory.Cohort`."""
    times, ratios, subtypes = {}, {}, {}
    for patno, g in cohort.finite.groupby("PATNO", sort=True):
        g = g.sort_values("finite_index")
        times[patno] = g["years_from_baseline"].to_numpy(dtype=float)
        ratios[patno] = g["ratio"].to_numpy(dtype=float)
        subtypes[patno] = g["subtype_name"].to_numpy()
    return FoldData(times=times, ratios=ratios, subtypes=subtypes)


# ---------------------------------------------------------------------------
#  Baselines
# ---------------------------------------------------------------------------
def _pooled_ols(sequences: list) -> np.ndarray:
    """Least-squares line through every training observation, ignoring patients.

    The population-only baseline. It is what you would predict knowing the
    cohort's average course and the date of the next visit, but nothing about
    the individual, so it isolates how much of any model's performance comes
    from the patient's own history.
    """
    t = np.concatenate([s["t"] for s in sequences])
    y = np.concatenate([s["y"] for s in sequences])
    X = np.column_stack([np.ones_like(t), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _baseline_residual_sd(sequences: list, beta: np.ndarray) -> dict:
    """Predictive spread for each baseline, from training patients only.

    Each baseline is applied to the training patients exactly as it will be
    applied to the held-out one --- one step ahead, using only what precedes the
    point --- and the spread of those errors becomes its predictive standard
    deviation. A baseline scored on coverage or Brier needs a distribution, and
    this is the only way to give it one without looking at the held-out patient.
    """
    errors = {name: [] for name in BASELINES}
    for seq in sequences:
        t, y = seq["t"], seq["y"]
        for k in range(1, len(y)):
            errors["persistence"].append(y[k] - y[k - 1])
            errors["patient_mean"].append(y[k] - y[:k].mean())
            errors["population"].append(y[k] - (beta[0] + beta[1] * t[k]))
            errors["patient_ols"].append(y[k] - _ols_extrapolate(t[:k], y[:k], t[k]))

    out = {}
    for name, e in errors.items():
        e = np.asarray(e, dtype=float)
        e = e[np.isfinite(e)]
        out[name] = float(np.sqrt(np.mean(e**2))) if len(e) else 1.0
    return out


def _ols_extrapolate(t_hist: np.ndarray, y_hist: np.ndarray, t_target: float) -> float:
    """Least-squares line through one patient's history, evaluated at the target.

    With a single observation there is no line to fit, so the value is carried
    forward; with two it interpolates them exactly, which is why this baseline
    is unstable early and is reported with its history length attached.
    """
    if len(t_hist) < 2 or np.ptp(t_hist) == 0:
        return float(y_hist[-1])
    X = np.column_stack([np.ones_like(t_hist), t_hist])
    beta, *_ = np.linalg.lstsq(X, y_hist, rcond=None)
    return float(beta[0] + beta[1] * t_target)


def _baseline_prediction(name: str, t_hist, y_hist, t_target, beta) -> float:
    if name == "persistence":
        return float(y_hist[-1])
    if name == "patient_mean":
        return float(np.mean(y_hist))
    if name == "population":
        return float(beta[0] + beta[1] * t_target)
    if name == "patient_ols":
        return _ols_extrapolate(t_hist, y_hist, t_target)
    raise ValueError(f"unknown baseline {name!r}")


# ---------------------------------------------------------------------------
#  One fold
# ---------------------------------------------------------------------------

def _kernel_spec(entry):
    """Normalise a kernel entry to (label, kernel, fit_mean).

    A plain string behaves exactly as before: `"K1"` means kernel K1 with the
    fitted linear mean, reported under the name "K1". A triple lets a variant
    of the same kernel be scored alongside it in the same fold --- for example
    `("K2_zeromean", "K2", False)` fits K2 with the mean fixed at zero and
    reports it as a separate model.

    Running both arms inside one fold is what makes the comparison paired: they
    see the identical training set, the identical held-out targets and the
    identical histories, so any difference is the mean specification and
    nothing else.
    """
    if isinstance(entry, str):
        return entry, entry, True
    label, kernel, fit_mean = entry
    return str(label), str(kernel), bool(fit_mean)


def _row(model, kernel, patno, k, t_hist, t_target, observed, subtype, mean, sd,
         n_train, converged, td, pigd) -> dict:
    probs = gpr.subtype_probabilities(mean, sd, td, pigd)
    return {
        "held_out_patient": patno,
        "model": model,
        "kernel": kernel,
        "target_index": k,
        "history_length": len(t_hist),
        "t_last_history": float(t_hist[-1]),
        "t_target": float(t_target),
        "horizon_years": float(t_target - t_hist[-1]),
        "observed_ratio": float(observed),
        "observed_subtype": subtype,
        "pred_mean": float(mean),
        "pred_sd": float(sd),
        "q2.5": float(mean - 1.959963985 * sd),
        "q97.5": float(mean + 1.959963985 * sd),
        "error": float(mean - observed),
        "negative_mass": gpr.negative_mass(mean, sd),
        **{f"p_{s}": probs[s] for s in STATE_NAMES},
        "n_train_patients": int(n_train),
        "train_converged": bool(converged),
    }


def run_fold(
    data: FoldData,
    patno,
    kernels: list,
    settings: dict,
    with_hbayes: bool = True,
    with_baselines: bool = True,
) -> tuple[list, list, dict]:
    """Fit every model without patient `patno`, then predict that patient.

    Returns (prediction rows, fit rows, fitted GPs). The fit rows carry the
    hyperparameters of every kernel on this fold, so a parameter that pins to a
    bound can be counted across folds rather than noticed once and forgotten.
    The fitted GPs are handed back so the multi-step forecast can reuse them:
    it conditions on a different slice of the same patient's history, which
    changes nothing about the training set they came from.
    """
    train = data.sequences(exclude=patno)
    if not train:
        raise ValueError(f"no training patients left after holding out {patno}")

    t_all = data.times[patno]
    y_all = data.ratios[patno]
    s_all = data.subtypes[patno]
    n_train = len(train)
    td = float(settings["td_threshold"])
    pigd = float(settings["pigd_threshold"])

    predictions: list = []
    fits: list = []
    fitted: dict = {}

    # ----------------------------------------------------------- GP kernels
    for label, kernel, fit_mean in (_kernel_spec(e) for e in kernels):
        fit = gpr.fit_gpr(
            train,
            kernel=kernel,
            n_restarts=int(settings["n_restarts"]),
            seed=int(settings["seed"]),
            max_iter=int(settings["max_iter"]),
            jitter=float(settings["jitter"]),
            fit_mean=fit_mean,
        )
        fitted[label] = fit
        fits.append(
            {
                "held_out_patient": patno,
                "kernel": label,
                "base_kernel": kernel,
                "fit_mean": fit_mean,
                "loglik": fit.log_marginal_likelihood,
                "beta0": float(fit.beta[0]),
                "beta1": float(fit.beta[1]),
                "converged": fit.converged,
                "at_boundary": ",".join(fit.at_boundary),
                "n_train_patients": n_train,
                **fit.params,
            }
        )
        for k in range(1, len(y_all)):
            mean, sd = gpr.predict_one(fit, t_all[:k], y_all[:k], float(t_all[k]),
                                       jitter=float(settings["jitter"]))
            predictions.append(
                _row(label, kernel, patno, k, t_all[:k], t_all[k], y_all[k],
                     s_all[k], mean, sd, n_train, fit.converged, td, pigd)
            )

    # ------------------------------------------- hierarchical growth model
    if with_hbayes:
        panel = pd.DataFrame(
            {
                "PATNO": np.concatenate([[i] * len(s["t"]) for i, s in enumerate(train)]),
                "years_from_baseline": np.concatenate([s["t"] for s in train]),
                "ratio": np.concatenate([s["y"] for s in train]),
            }
        )
        growth = hbayes.build_growth_data(panel)
        post = hbayes.fit_growth_model(
            growth,
            n_draws=int(settings["hbayes_draws"]),
            burn_in=int(settings["hbayes_burn_in"]),
            seed=int(settings["seed"]),
        )
        for k in range(1, len(y_all)):
            draws = hbayes.predict_ratio(
                post, t_all[:k], y_all[:k], float(t_all[k]),
                seed=int(settings["seed"]) + k,
                max_draws=int(settings["hbayes_pred_draws"]),
            )
            predictions.append(
                _row("hbayes", "", patno, k, t_all[:k], t_all[k], y_all[k],
                     s_all[k], float(draws.mean()), float(draws.std()),
                     n_train, True, td, pigd)
            )

    # --------------------------------------------------------- baselines
    if with_baselines:
        beta = _pooled_ols(train)
        sds = _baseline_residual_sd(train, beta)
        for name in BASELINES:
            for k in range(1, len(y_all)):
                mean = _baseline_prediction(name, t_all[:k], y_all[:k],
                                            float(t_all[k]), beta)
                predictions.append(
                    _row(name, "", patno, k, t_all[:k], t_all[k], y_all[k],
                         s_all[k], mean, sds[name], n_train, True, td, pigd)
                )

    return predictions, fits, fitted


def run_multistep_fold(data: FoldData, patno, kernels: list, settings: dict,
                       min_finite: int, fitted: dict | None = None) -> list:
    """Forecast the back half of one patient's trajectory from the front half.

    The joint predictive covariance over the forecast points is used, not just
    the pointwise variances: `traj_loglik_joint` scores the whole observed
    future trajectory as one multivariate draw, which is what a *trajectory*
    forecast should be judged on. Scoring each point separately would treat
    correlated errors as independent evidence.
    """
    t_all, y_all = data.times[patno], data.ratios[patno]
    s_all = data.subtypes[patno]
    T = len(y_all)
    if T < min_finite:
        return []
    m = int(np.ceil(T / 2))
    if m >= T:
        return []

    td, pigd = float(settings["td_threshold"]), float(settings["pigd_threshold"])
    rows = []

    for label, kernel, fit_mean in (_kernel_spec(e) for e in kernels):
        fit = (fitted or {}).get(label)
        if fit is None:
            fit = gpr.fit_gpr(
                data.sequences(exclude=patno), kernel=kernel,
                n_restarts=int(settings["n_restarts"]), seed=int(settings["seed"]),
                max_iter=int(settings["max_iter"]), jitter=float(settings["jitter"]),
                fit_mean=fit_mean,
            )
        elif fit.kernel != kernel:
            raise ValueError(f"fitted GP for {label!r} carries kernel {fit.kernel!r}")
        mean, cov = gpr.posterior(fit, t_all[:m], y_all[:m], t_all[m:],
                                  jitter=float(settings["jitter"]))
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        observed = y_all[m:]
        resid = observed - mean
        C = cov + float(settings["jitter"]) * np.eye(len(mean))
        sign, logdet = np.linalg.slogdet(C)
        joint = (
            float(-0.5 * (resid @ np.linalg.solve(C, resid) + logdet
                          + len(mean) * np.log(2 * np.pi)))
            if sign > 0 else np.nan
        )

        for j, k in enumerate(range(m, T)):
            probs = gpr.subtype_probabilities(mean[j], sd[j], td, pigd)
            rows.append(
                {
                    "held_out_patient": patno,
                    "model": label,
                    "history_length": m,
                    "step": j + 1,
                    "target_index": k,
                    "t_last_history": float(t_all[m - 1]),
                    "t_target": float(t_all[k]),
                    "horizon_years": float(t_all[k] - t_all[m - 1]),
                    "observed_ratio": float(observed[j]),
                    "observed_subtype": s_all[k],
                    "pred_mean": float(mean[j]),
                    "pred_sd": float(sd[j]),
                    "error": float(mean[j] - observed[j]),
                    **{f"p_{s}": probs[s] for s in STATE_NAMES},
                    "n_forecast_points": int(T - m),
                    "traj_loglik_joint": joint,
                }
            )
    return rows


# ---------------------------------------------------------------------------
#  The loop
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(data, kernels, settings, with_hbayes, with_baselines, multistep,
                 min_finite):
    _WORKER.update(
        data=data, kernels=kernels, settings=settings, with_hbayes=with_hbayes,
        with_baselines=with_baselines, multistep=multistep, min_finite=min_finite,
    )
    # Each worker owns one core; threaded BLAS on matrices this small only
    # costs time, and oversubscribing across workers costs more.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def _run_one(patno):
    w = _WORKER
    predictions, fits, fitted = run_fold(
        w["data"], patno, w["kernels"], w["settings"],
        with_hbayes=w["with_hbayes"], with_baselines=w["with_baselines"],
    )
    multistep = (
        run_multistep_fold(w["data"], patno, w["kernels"], w["settings"],
                           w["min_finite"], fitted=fitted)
        if w["multistep"] else []
    )
    return predictions, fits, multistep


def run_lopo(
    data: FoldData,
    kernels: list,
    settings: dict,
    patients: list | None = None,
    with_hbayes: bool = True,
    with_baselines: bool = True,
    multistep: bool = True,
    min_finite: int = 3,
    jobs: int = 1,
    progress=None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave one patient out, over the whole cohort.

    Folds are independent, so `jobs > 1` spreads them across processes without
    changing a single number: each fold refits from the same seed on the same
    training set regardless of which worker runs it.

    With `checkpoint`, completed folds are flushed to disk every
    `checkpoint_every` folds and any already present there are skipped on a
    later call. A cohort takes hours, and without this an interruption at fold
    640 of 641 loses every one of them --- which is exactly what happened to the
    first D1 run.

    Resuming is safe because a fold depends only on the cohort and the seed,
    never on which folds ran before it. The checkpoint is keyed by held-out
    patient, so a partially written fold is discarded rather than half-counted.
    """
    patients = list(data.patients) if patients is None else list(patients)
    unknown = [p for p in patients if p not in data.times]
    if unknown:
        raise ValueError(f"{len(unknown)} requested patient(s) are not in the cohort")

    preds, fits, steps = [], [], []
    done = 0

    # ------------------------------------------------------------- resume
    resumed: list = []
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        existing = _load_checkpoint(checkpoint)
        if existing is not None:
            complete = set(existing[0]["held_out_patient"].unique())
            # Only skip patients this call was asked for; a checkpoint from a
            # larger cohort must not silently shrink the requested work.
            resumed = [p for p in patients if p in complete]
            if resumed:
                for frame, bucket in zip(existing, (preds, fits, steps)):
                    if not frame.empty:
                        keep = frame[frame["held_out_patient"].isin(resumed)]
                        bucket.extend(keep.to_dict("records"))
                patients = [p for p in patients if p not in complete]
                done = len(resumed)

    def flush():
        if checkpoint is not None:
            _write_checkpoint(checkpoint, preds, fits, steps)

    if jobs > 1:
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker,
            initargs=(data, kernels, settings, with_hbayes, with_baselines,
                      multistep, min_finite),
        ) as pool:
            for p, f, s in pool.map(_run_one, patients, chunksize=1):
                preds.extend(p)
                fits.extend(f)
                steps.extend(s)
                done += 1
                if checkpoint is not None and done % checkpoint_every == 0:
                    flush()
                if progress:
                    progress(done, len(patients) + len(resumed))
    else:
        _init_worker(data, kernels, settings, with_hbayes, with_baselines,
                     multistep, min_finite)
        for patno in patients:
            p, f, s = _run_one(patno)
            preds.extend(p)
            fits.extend(f)
            steps.extend(s)
            done += 1
            if checkpoint is not None and done % checkpoint_every == 0:
                flush()
            if progress:
                progress(done, len(patients) + len(resumed))

    flush()
    return pd.DataFrame(preds), pd.DataFrame(fits), pd.DataFrame(steps)


def _checkpoint_paths(checkpoint: Path) -> tuple:
    stem = checkpoint.with_suffix("")
    return (Path(f"{stem}_predictions.csv"), Path(f"{stem}_fits.csv"),
            Path(f"{stem}_multistep.csv"))


def _load_checkpoint(checkpoint: Path):
    """Read a checkpoint, or None if it is absent or unusable.

    A truncated file from a process killed mid-write is treated as absent: the
    folds are cheap to recompute relative to the risk of resuming from a
    half-written row.
    """
    paths = _checkpoint_paths(checkpoint)
    if not paths[0].exists():
        return None
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p) if p.exists() else pd.DataFrame())
        except Exception:                                     # noqa: BLE001
            return None
    if frames[0].empty or "held_out_patient" not in frames[0]:
        return None
    return frames


def _write_checkpoint(checkpoint: Path, preds, fits, steps) -> None:
    """Write the checkpoint atomically, so an interrupt cannot truncate it."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(_checkpoint_paths(checkpoint), (preds, fits, steps)):
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)


# ---------------------------------------------------------------------------
#  Checks
# ---------------------------------------------------------------------------
def gaussian_crps(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score for a Gaussian predictive.

        CRPS = sd * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],   z = (y - mu)/sd

    A proper score that reads on the scale of the ratio itself, so it can be
    compared with MAE directly; unlike MAE it rewards a prediction for being
    appropriately uncertain rather than only for being close.
    """
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def continuous_metrics(df: pd.DataFrame) -> dict:
    """How close the predicted ratio is, and how honest its uncertainty is.

    These are the headline numbers for Step 3. The categorical metrics below
    are derived quantities and are reported second, because collapsing a
    predictive distribution onto three bands discards most of what was
    predicted.
    """
    y = df["observed_ratio"].to_numpy(dtype=float)
    mu = df["pred_mean"].to_numpy(dtype=float)
    sd = df["pred_sd"].to_numpy(dtype=float)
    err = mu - y
    inside = (y >= df["q2.5"].to_numpy()) & (y <= df["q97.5"].to_numpy())

    return {
        "n": int(len(df)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedAE": float(np.median(np.abs(err))),
        "bias": float(np.mean(err)),
        "coverage95": float(np.mean(inside)),
        "interval_width": float(np.mean(df["q97.5"] - df["q2.5"])),
        "CRPS": float(np.mean(gaussian_crps(y, mu, sd))),
        "loglik_ratio": float(np.mean(norm.logpdf(y, loc=mu, scale=np.maximum(sd, 1e-12)))),
        "corr": float(np.corrcoef(mu, y)[0, 1]) if len(df) > 2 else np.nan,
        "negative_mass": float(df["negative_mass"].mean())
        if "negative_mass" in df else np.nan,
    }


def categorical_metrics(df: pd.DataFrame) -> dict:
    """Subtype performance derived from the predictive distribution.

    Secondary by design. The Indeterminate band is 0.25 wide against a
    predictive spread several times that, so no model of this family can place
    much mass inside it; a low Indeterminate recall is a fact about the band,
    not a verdict on the model.
    """
    probs = np.clip(df[PROB_COLS].to_numpy(dtype=float), 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    truth = df["observed_subtype"].map({s: i for i, s in enumerate(STATE_NAMES)}).to_numpy()

    predicted = probs.argmax(axis=1)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(truth)), truth] = 1.0

    recalls = [
        float((predicted[truth == i] == i).mean()) if (truth == i).any() else np.nan
        for i in range(len(STATE_NAMES))
    ]
    return {
        "n": int(len(df)),
        "accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "loglik_subtype": float(np.mean(np.log(probs[np.arange(len(truth)), truth]))),
        **{f"recall_{s}": r for s, r in zip(STATE_NAMES, recalls)},
    }


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Rows predicted, columns actual --- the orientation used elsewhere here."""
    probs = df[PROB_COLS].to_numpy(dtype=float)
    predicted = [STATE_NAMES[i] for i in probs.argmax(axis=1)]
    table = pd.crosstab(
        pd.Series(predicted, name="predicted"),
        df["observed_subtype"].rename("actual"),
    )
    return table.reindex(index=STATE_NAMES, columns=STATE_NAMES, fill_value=0)


def metrics_table(predictions: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Continuous and categorical metrics for every model, one row each."""
    rows = []
    for model, g in predictions.groupby("model", sort=False):
        rows.append(
            {"cohort": cohort, "model": model,
             **continuous_metrics(g), **categorical_metrics(g)}
        )
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def stratified_metrics(predictions: pd.DataFrame, cohort: str, by: str,
                       bins=None) -> pd.DataFrame:
    """Metrics split by history length or prediction horizon.

    A single MAE hides whether a model is helped by a longer history or hurt by
    a longer horizon, which is most of what a trajectory claim rests on.
    """
    df = predictions.copy()
    if bins is not None:
        df["_stratum"] = pd.cut(df[by], bins)
    else:
        df["_stratum"] = df[by].clip(upper=8)

    rows = []
    for (model, stratum), g in df.groupby(["model", "_stratum"], observed=True):
        if len(g) < 5:
            continue
        rows.append(
            {"cohort": cohort, "model": model, "by": by, "stratum": str(stratum),
             **continuous_metrics(g)}
        )
    return pd.DataFrame(rows)


def trajectory_metrics(steps: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Multi-step forecast quality, scored per patient then averaged.

    Averaging within a patient first stops the handful of patients with long
    trajectories from dominating: a patient with seven forecast points would
    otherwise count seven times as much as one with a single point.
    """
    if steps.empty:
        return pd.DataFrame()

    rows = []
    for model, g in steps.groupby("model", sort=False):
        per_patient = []
        for patno, p in g.groupby("held_out_patient"):
            obs = p["observed_ratio"].to_numpy(dtype=float)
            pred = p["pred_mean"].to_numpy(dtype=float)
            entry = {
                "mad": float(np.mean(np.abs(pred - obs))),
                "joint_loglik": float(p["traj_loglik_joint"].iloc[0]),
            }
            if len(obs) > 2 and np.std(obs) > 0 and np.std(pred) > 0:
                entry["corr"] = float(np.corrcoef(pred, obs)[0, 1])
            if len(obs) >= 2:
                # Did the trajectory move in the direction the model expected?
                entry["direction"] = float(
                    np.sign(obs[-1] - obs[0]) == np.sign(pred[-1] - pred[0])
                )
            per_patient.append(entry)

        frame = pd.DataFrame(per_patient)
        rows.append(
            {
                "cohort": cohort,
                "model": model,
                "n_patients": int(g["held_out_patient"].nunique()),
                "n_points": int(len(g)),
                "trajectory_MAD": float(frame["mad"].mean()),
                "trajectory_corr": float(frame["corr"].mean()) if "corr" in frame else np.nan,
                "direction_agreement": float(frame["direction"].mean())
                if "direction" in frame else np.nan,
                "joint_loglik": float(frame["joint_loglik"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("trajectory_MAD").reset_index(drop=True)


def check_predictions(predictions: pd.DataFrame, data: FoldData, cohort_name: str) -> list:
    """Leakage and consistency checks, run before any number is reported.

    Returns a list of (description, passed) pairs so the driver can print the
    whole battery rather than stopping at the first failure.
    """
    checks = []

    def add(label, ok):
        checks.append((label, bool(ok)))

    expected_points = sum(len(v) - 1 for v in data.ratios.values())
    per_model = predictions.groupby("model").size()
    add("every model scored every prediction point",
        bool((per_model == expected_points).all()))

    add("held-out patient is the patient being predicted",
        set(predictions["held_out_patient"]) <= set(data.patients))

    n_patients = len(data.patients)
    add("every fold trained on all other patients",
        bool((predictions["n_train_patients"] == n_patients - 1).all()))

    add("history is strictly the visits before the target",
        bool((predictions["history_length"] == predictions["target_index"]).all()))

    add("target is later than the end of its history",
        bool((predictions["t_target"] >= predictions["t_last_history"]).all()))

    add("no prediction has an empty history",
        bool((predictions["history_length"] >= 1).all()))

    probs = predictions[PROB_COLS].to_numpy(dtype=float)
    add("subtype probabilities sum to one",
        bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)))

    add("predictive standard deviations are positive",
        bool((predictions["pred_sd"] > 0).all()))

    add("no predicted or observed value is missing",
        bool(predictions[["pred_mean", "pred_sd", "observed_ratio"]].notna().all().all()))

    observed = predictions.groupby(["held_out_patient", "target_index"])["observed_ratio"].nunique()
    add("each (patient, visit) has one observed value across models",
        bool((observed == 1).all()))

    return checks
