"""Scoring held-out predictions: proper scores, calibration, paired bootstrap.

Every comparison in the cluster-phase experiments is made on *identical
held-out points* from *identical patient-level folds*, and every interval is a
patient-cluster bootstrap in which **all models and all comparisons share the
same resampled patients**. Sharing the draws is what makes a paired difference
paired: two models are judged on the same pseudo-cohort in every replicate.

The bootstrap is computed on per-patient sums, not on rows. A replicate draws
patients with replacement, which is a multinomial count vector ``w`` over
patients; any pooled mean is then ``(w @ S) / (w @ n)`` with ``S`` the
per-patient sum of the quantity and ``n`` the per-patient point count. That is
exactly the row-level resampling estimator, computed without materialising the
rows, so 4,000 replicates over 934 patients cost a few milliseconds.

Continuous scores (all for a Gaussian predictive N(mu, sd^2))
------------------------------------------------------------
* absolute and squared error of the predictive mean
* log predictive density  log N(y; mu, sd^2)            (higher is better)
* CRPS, closed form for a Gaussian                      (lower is better)
* PIT  u = Phi((y - mu) / sd); calibrated forecasts give u ~ U(0, 1)
* central-interval coverage at several nominal levels
* predictive mass below zero, where no subscore can lie

The subscores have a point mass at zero and are bounded, so a Gaussian
predictive is misspecified there. These scores are reported *as computed on the
Gaussian*, never after clipping, so the misspecification is visible rather than
hidden; the negative-mass column measures it directly.

Categorical scores
------------------
accuracy, balanced accuracy, per-class recall and precision, multiclass log
loss, Brier score, and a reliability table per class. Log loss and Brier are
rule-free: they score the probabilities, not the argmax, which matters here
because an argmax rule can never choose a class whose probability is never the
largest (Indeterminate), however informative those probabilities are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

CLASSES = ("TD", "Indeterminate", "PIGD")
LEVELS = (0.5, 0.8, 0.9, 0.95)
_LOG_EPS = 1e-12


# ---------------------------------------------------------------------------
#  Gaussian scores
# ---------------------------------------------------------------------------
def gaussian_log_score(y, mu, sd):
    """log N(y; mu, sd^2), elementwise."""
    y, mu, sd = map(np.asarray, (y, mu, sd))
    return norm.logpdf(y, loc=mu, scale=sd)


def gaussian_crps(y, mu, sd):
    """Closed-form CRPS of N(mu, sd^2) at y (Gneiting & Raftery 2007, eq. 21).

        CRPS = sd * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sd
    """
    y, mu, sd = map(np.asarray, (y, mu, sd))
    z = (y - mu) / sd
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def pit(y, mu, sd):
    y, mu, sd = map(np.asarray, (y, mu, sd))
    return norm.cdf((y - mu) / sd)


def covered(y, mu, sd, level):
    """Whether y lies in the central `level` predictive interval."""
    z = norm.ppf(0.5 + level / 2.0)
    return np.abs(np.asarray(y) - np.asarray(mu)) <= z * np.asarray(sd)


def negative_mass(mu, sd):
    return norm.cdf(0.0, loc=np.asarray(mu), scale=np.asarray(sd))


def pointwise_continuous(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Per-point scores for one outcome; columns obs_/pred_/sd_<outcome> required."""
    y = frame[f"obs_{outcome}"].to_numpy(float)
    mu = frame[f"pred_{outcome}"].to_numpy(float)
    sd = frame[f"sd_{outcome}"].to_numpy(float)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(f"non-positive or non-finite predictive SD for {outcome}")
    out = {
        "ae": np.abs(mu - y),
        "se": (mu - y) ** 2,
        "log_score": gaussian_log_score(y, mu, sd),
        "crps": gaussian_crps(y, mu, sd),
        "pit": pit(y, mu, sd),
        "neg_mass": negative_mass(mu, sd),
        "bias": mu - y,
    }
    for lv in LEVELS:
        out[f"cov{int(round(lv * 100))}"] = covered(y, mu, sd, lv).astype(float)
    return pd.DataFrame(out, index=frame.index)


# ---------------------------------------------------------------------------
#  Patient-cluster bootstrap with shared draws
# ---------------------------------------------------------------------------
@dataclass
class SharedBootstrap:
    """Multinomial patient weights reused by every estimate and comparison."""

    patients: np.ndarray        # sorted unique patient ids
    weights: np.ndarray         # (n_boot, n_patients) resample counts
    seed: int

    @classmethod
    def create(cls, patients, n_boot: int = 4000, seed: int = 42) -> "SharedBootstrap":
        uniq = np.unique(np.asarray(patients))
        rng = np.random.default_rng(seed)
        P = len(uniq)
        w = rng.multinomial(P, np.full(P, 1.0 / P), size=n_boot).astype(float)
        return cls(patients=uniq, weights=w, seed=seed)

    def _sums(self, values: np.ndarray, patients: np.ndarray):
        pos = np.searchsorted(self.patients, patients)
        if np.any(self.patients[np.clip(pos, 0, len(self.patients) - 1)] != patients):
            raise ValueError("a patient is not in the bootstrap frame")
        S = np.bincount(pos, weights=values, minlength=len(self.patients))
        n = np.bincount(pos, minlength=len(self.patients)).astype(float)
        return S, n

    def mean_ci(self, values, patients, alpha: float = 0.05):
        """Pooled mean and percentile interval; the estimand weights points equally."""
        values = np.asarray(values, float)
        patients = np.asarray(patients)
        S, n = self._sums(values, patients)
        draws = (self.weights @ S) / np.maximum(self.weights @ n, 1e-300)
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(values.mean()), float(lo), float(hi)

    def ratio_ci(self, num, den, patients, alpha: float = 0.05):
        """Ratio of pooled sums, e.g. a recall = correct / support within a class."""
        num, den = np.asarray(num, float), np.asarray(den, float)
        patients = np.asarray(patients)
        Sn, _ = self._sums(num, patients)
        Sd, _ = self._sums(den, patients)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (self.weights @ Sn) / (self.weights @ Sd)
        draws = draws[np.isfinite(draws)]
        est = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
        if not len(draws):
            return est, float("nan"), float("nan")
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return est, float(lo), float(hi)


# ---------------------------------------------------------------------------
#  Subtype from a bivariate Gaussian predictive
# ---------------------------------------------------------------------------
def classify_draws(tremor: np.ndarray, pigd: np.ndarray, td: float, pg: float):
    """Apply the project's 3-class rule to (clipped) draws; returns 0/1/2 codes.

    Matches `subtype.classify` with ``band_boundaries: outer``:
        PIGD == 0 and tremor > 0  -> TD
        PIGD == 0 and tremor == 0 -> Indeterminate
        ratio >= td               -> TD
        ratio <= pg               -> PIGD
        otherwise                 -> Indeterminate
    """
    ratio = np.divide(tremor, pigd, out=np.full_like(tremor, np.inf), where=pigd > 0)
    is_td = ((pigd > 0) & (ratio >= td)) | ((pigd == 0) & (tremor > 0))
    is_pigd = (pigd > 0) & (ratio <= pg)
    return np.where(is_td, 0, np.where(is_pigd, 2, 1))


def subtype_probabilities(mu_t, mu_p, sd_t, sd_p, rho, td, pg,
                          n_draws: int = 4000, seed: int = 42):
    """Monte Carlo class probabilities from N2((mu_t, mu_p), cov(rho)).

    Draws are clipped at zero (a subscore cannot be negative) before the rule is
    applied; the clipped mass is returned so it is reported, not hidden. The
    same standard-normal draws are used for every row and every model (fixed
    seed), so differences between models are not Monte Carlo noise.
    """
    mu_t, mu_p, sd_t, sd_p, rho = (np.asarray(a, float)[:, None]
                                   for a in (mu_t, mu_p, sd_t, sd_p, rho))
    rho = np.clip(rho, -0.999, 0.999)
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((1, n_draws))
    z2 = rng.standard_normal((1, n_draws))
    raw_t = mu_t + sd_t * z1
    raw_p = mu_p + sd_p * (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2)
    clipped = ((raw_t < 0) | (raw_p < 0)).mean(axis=1)
    codes = classify_draws(np.clip(raw_t, 0, None), np.clip(raw_p, 0, None), td, pg)
    probs = np.stack([(codes == k).mean(axis=1) for k in range(3)], axis=1)
    return probs, clipped


def point_class(mu_t, mu_p, td, pg):
    """The rule applied to the predictive means (clipped at zero)."""
    t = np.clip(np.asarray(mu_t, float), 0, None)
    p = np.clip(np.asarray(mu_p, float), 0, None)
    return classify_draws(t, p, td, pg)


def categorical_pointwise(probs: np.ndarray, actual: np.ndarray, pred: np.ndarray):
    """Per-point log loss, Brier and correctness. `actual`/`pred` are 0/1/2 codes."""
    n = len(actual)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual] = 1.0
    p_true = np.clip(probs[np.arange(n), actual], _LOG_EPS, None)
    return {
        "log_loss": -np.log(p_true),
        "brier": ((probs - onehot) ** 2).sum(axis=1),
        "correct": (pred == actual).astype(float),
    }


def categorical_summary(probs, actual, pred, patients, boot: SharedBootstrap) -> dict:
    pw = categorical_pointwise(probs, actual, pred)
    out = {"n": int(len(actual))}
    for k in ("correct", "log_loss", "brier"):
        name = "accuracy" if k == "correct" else k
        est, lo, hi = boot.mean_ci(pw[k], patients)
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = est, lo, hi
    recalls = []
    for c, cname in enumerate(CLASSES):
        support = (actual == c).astype(float)
        hit = ((actual == c) & (pred == c)).astype(float)
        est, lo, hi = boot.ratio_ci(hit, support, patients)
        out[f"recall_{cname}"], out[f"recall_{cname}_lo"], out[f"recall_{cname}_hi"] = est, lo, hi
        npred = int((pred == c).sum())
        out[f"n_predicted_{cname}"] = npred
        out[f"precision_{cname}"] = (float(((actual == c) & (pred == c)).sum() / npred)
                                     if npred else float("nan"))
        out[f"support_{cname}"] = int(support.sum())
        out[f"mean_prob_{cname}"] = float(probs[:, c].mean())
        recalls.append(est)
    out["balanced_accuracy"] = float(np.nanmean(recalls))
    # Balanced accuracy CI from the same shared draws.
    pos = np.searchsorted(boot.patients, np.asarray(patients))
    per_draw = []
    for c in range(3):
        Sh = np.bincount(pos, weights=((actual == c) & (pred == c)).astype(float),
                         minlength=len(boot.patients))
        Ss = np.bincount(pos, weights=(actual == c).astype(float),
                         minlength=len(boot.patients))
        with np.errstate(invalid="ignore", divide="ignore"):
            per_draw.append((boot.weights @ Sh) / (boot.weights @ Ss))
    ba = np.nanmean(np.stack(per_draw), axis=0)
    out["balanced_accuracy_lo"], out["balanced_accuracy_hi"] = (
        float(np.nanpercentile(ba, 2.5)), float(np.nanpercentile(ba, 97.5)))
    return out


def reliability_table(probs, actual, n_bins: int = 10) -> pd.DataFrame:
    """Per-class reliability: mean predicted probability vs observed frequency."""
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    for c, cname in enumerate(CLASSES):
        p = probs[:, c]
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.any():
                rows.append({"class": cname, "bin_low": edges[b], "bin_high": edges[b + 1],
                             "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                             "observed_rate": float((actual[m] == c).mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(probs, actual, n_bins: int = 10) -> dict:
    tab = reliability_table(probs, actual, n_bins)
    out = {}
    for cname, g in tab.groupby("class"):
        out[f"ece_{cname}"] = float((g.n * (g.mean_predicted - g.observed_rate).abs()).sum()
                                    / g.n.sum())
    return out
