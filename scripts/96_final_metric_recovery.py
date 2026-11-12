"""Recover every classification metric that the saved prediction files support.

Nothing here fits a model. Every number is computed from predictions that an
earlier experiment already wrote to disk, on that experiment's own evaluation
cohort, with PIGD as the positive class for the binary task.

Usage:
    python scripts/96_final_metric_recovery.py --out experiments/<run>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
LABELS = ("TD", "Indeterminate", "PIGD")
KEY = ["PATNO", "current_event", "target_event"]
N_BOOT = 2000
SEED = 42
#: the repository clips the true-class probability here before taking a log
#: (src/ppmi_subtype/next_visit.py:118, src/ppmi_subtype/two_state_hmm.py:145),
#: so the same constant is used to keep recovered values comparable.
LOGLOSS_CLIP = 1e-12

# ---------------------------------------------------------------- sources

RUNS = {
    "profile": E / "20261103T220147Z_a2p_outcome_table_fullprec",
    "lopo": E / "20261020T104756Z_next_visit_td_pigd_prediction",
    "band": E / "20261105T025024Z_cthmm_ratio_band_width_extended",
    "band_first": E / "20261104T152154Z_cthmm_ratio_banded_observations",
    "two_state": E / "20261110T143200Z_two_state_hmm_confirm",
    "harmonised": E / "20261103T202632Z_e8_harmonised_family_comparison",
    "auc": E / "20261023T202640Z_roc_auc_comparison",
    "ic": E / "20261023T194317Z_model_selection_criteria",
    "profile_cont": E / "20261103T213859Z_a2_clinical_profile_cv_rerun",
    "no_ledd": E / "20261111T121021Z_rich_profile_no_ledd",
    "band_diag": E / "20261104T155439Z_ratio_band_emission_diagnostic",
}


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- metrics


def fast_auc(pos: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with mid-ranks for ties."""
    n1 = int(pos.sum())
    n0 = pos.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fast_ap(pos: np.ndarray, score: np.ndarray) -> float:
    """Average precision, thresholded at distinct scores so ties are grouped.

    Equal scores cannot be ordered, so precision and recall are evaluated once
    per distinct score rather than once per row; this is what
    sklearn.metrics.average_precision_score does and it matters here because
    several arms emit rule-based probabilities that take only two values.
    """
    n1 = int(pos.sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = pos[order]
    s = score[order]
    tp = np.cumsum(y)
    k = np.arange(1, y.size + 1)
    last = np.flatnonzero(np.diff(s) != 0)
    last = np.concatenate((last, [y.size - 1]))
    prec = tp[last] / k[last]
    rec = tp[last] / n1
    return float(np.sum(np.diff(np.concatenate(([0.0], rec))) * prec))


def calibration(pos: np.ndarray, score: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration slope, intercept and binned error.

    The slope and intercept come from a logistic regression of the realised
    outcome on the logit of the model's probability, fitted by Newton steps;
    a perfectly calibrated model gives slope 1 and intercept 0. A slope below
    1 means the probabilities are too extreme, above 1 too timid. Both are
    undefined when the score takes fewer than three distinct values, and are
    returned as nan rather than a number in that case.
    """
    y = pos.astype(float)
    out = {
        "calibration_in_the_large": float(np.mean(score) - np.mean(y)),
        "calibration_slope": float("nan"),
        "calibration_intercept": float("nan"),
    }
    # expected and maximum calibration error over equal-width probability bins
    idx = np.clip((score * n_bins).astype(int), 0, n_bins - 1)
    gaps, weights = [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gaps.append(abs(float(score[m].mean() - y[m].mean())))
        weights.append(int(m.sum()))
    w = np.asarray(weights, dtype=float)
    g = np.asarray(gaps, dtype=float)
    out["ece_10_bin"] = float(np.sum(w * g) / w.sum())
    out["mce_10_bin"] = float(g.max())
    out["n_calibration_bins_used"] = int(w.size)

    p = np.clip(score, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    if np.unique(x).size < 3:
        return out
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(100):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-eta))
        wgt = np.maximum(mu * (1.0 - mu), 1e-10)
        z = eta + (y - mu) / wgt
        XtW = X.T * wgt
        try:
            nb = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return out
        if not np.all(np.isfinite(nb)):
            return out
        step = float(np.max(np.abs(nb - b)))
        b = nb
        if step < 1e-10:
            break
    out["calibration_intercept"] = float(b[0])
    out["calibration_slope"] = float(b[1])
    return out


def binary_core(pos: np.ndarray, score: np.ndarray, called: np.ndarray) -> dict:
    """pos/called are boolean (True = PIGD); score is P(PIGD)."""
    tp = int(np.sum(pos & called))
    fn = int(np.sum(pos & ~called))
    fp = int(np.sum(~pos & called))
    tn = int(np.sum(~pos & ~called))
    n = tp + fn + fp + tn
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0:
        f1 = 2 * prec * sens / (prec + sens)
    elif tp + fn > 0:
        f1 = 0.0  # the class has support but the model never predicts it
    else:
        f1 = float("nan")
    y = pos.astype(float)
    p_true_raw = np.where(pos, score, 1.0 - score)
    p_true = np.clip(p_true_raw, LOGLOSS_CLIP, 1.0)
    out = {
        "n_scored": n,
        "n_positive_PIGD": tp + fn,
        "n_negative_TD": tn + fp,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "sensitivity_PIGD": sens,
        "specificity_TD": spec,
        "precision_PIGD": prec,
        "npv_TD": npv,
        "f1_PIGD": f1,
        "balanced_accuracy": (sens + spec) / 2.0,
        "auc_roc": fast_auc(pos, score),
        "pr_auc": fast_ap(pos, score),
        "brier_binary": float(np.mean((score - y) ** 2)),
        "log_loss": float(-np.mean(np.log(p_true))),
        "log_loss_clipping_active": bool(np.any(p_true_raw < LOGLOSS_CLIP)),
        "n_zero_prob_on_true_class": int(np.sum(p_true_raw <= 0.0)),
    }
    out.update(calibration(pos, score))
    return out


BIN_CI = (
    "accuracy",
    "sensitivity_PIGD",
    "specificity_TD",
    "precision_PIGD",
    "f1_PIGD",
    "balanced_accuracy",
    "auc_roc",
    "pr_auc",
    "brier_binary",
    "log_loss",
    "calibration_slope",
    "calibration_intercept",
    "ece_10_bin",
    "calibration_in_the_large",
)


def three_class_core(y_idx: np.ndarray, P: np.ndarray, pred_idx: np.ndarray) -> dict:
    out: dict = {}
    n = y_idx.size
    out["n_scored"] = n
    out["accuracy"] = float(np.mean(pred_idx == y_idx))
    f1s, recalls, supports, aucs, aps = [], [], [], [], []
    for k, name in enumerate(LABELS):
        pos = y_idx == k
        called = pred_idx == k
        tp = int(np.sum(pos & called))
        fn = int(np.sum(pos & ~called))
        fp = int(np.sum(~pos & called))
        tn = int(np.sum(~pos & ~called))
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        f1 = (
            2 * prec * sens / (prec + sens)
            if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0
            else 0.0
        )
        auc = fast_auc(pos, P[:, k])
        ap = fast_ap(pos, P[:, k])
        out[f"support_{name}"] = tp + fn
        out[f"n_predicted_{name}"] = tp + fp
        out[f"sensitivity_{name}"] = sens
        out[f"specificity_{name}"] = spec
        out[f"precision_{name}"] = prec
        out[f"f1_{name}"] = f1
        out[f"auc_ovr_{name}"] = auc
        out[f"pr_auc_ovr_{name}"] = ap
        out[f"n_distinct_scores_{name}"] = int(np.unique(P[:, k]).size)
        f1s.append(f1)
        recalls.append(sens)
        supports.append(tp + fn)
        aucs.append(auc)
        aps.append(ap)
    supports = np.asarray(supports, dtype=float)
    w = supports / supports.sum()
    out["macro_f1"] = float(np.mean(f1s))
    out["weighted_f1"] = float(np.sum(w * np.asarray(f1s)))
    out["balanced_accuracy"] = float(np.nanmean(recalls))
    out["macro_auc_ovr"] = float(np.nanmean(aucs))
    out["weighted_auc_ovr"] = float(np.nansum(w * np.asarray(aucs)))
    out["macro_pr_auc_ovr"] = float(np.nanmean(aps))
    Y = np.zeros_like(P)
    Y[np.arange(n), y_idx] = 1.0
    out["brier_multiclass_summed"] = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    out["brier_multiclass_mean_over_classes"] = out["brier_multiclass_summed"] / 3.0
    pt_raw = P[np.arange(n), y_idx]
    pt = np.clip(pt_raw, LOGLOSS_CLIP, 1.0)
    out["log_loss"] = float(-np.mean(np.log(pt)))
    out["log_loss_clipping_active"] = bool(np.any(pt_raw < LOGLOSS_CLIP))
    out["n_zero_prob_on_true_class"] = int(np.sum(pt_raw <= 0.0))
    out["mean_predictive_loglik"] = -out["log_loss"]
    out["sum_predictive_loglik"] = float(-out["log_loss"] * n)
    return out


MC_CI = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "sensitivity_TD",
    "sensitivity_Indeterminate",
    "sensitivity_PIGD",
    "specificity_TD",
    "specificity_Indeterminate",
    "specificity_PIGD",
    "f1_TD",
    "f1_Indeterminate",
    "f1_PIGD",
    "macro_auc_ovr",
    "auc_ovr_TD",
    "auc_ovr_Indeterminate",
    "auc_ovr_PIGD",
    "pr_auc_ovr_TD",
    "pr_auc_ovr_Indeterminate",
    "pr_auc_ovr_PIGD",
    "brier_multiclass_summed",
    "log_loss",
    "mean_predictive_loglik",
)


# ------------------------------------------------------- patient bootstrap


@dataclass
class Boot:
    """Patient-level resampling index, built once per cohort."""

    groups: list[np.ndarray]
    rng: np.random.Generator

    @classmethod
    def build(cls, patients: np.ndarray, seed: int = SEED) -> "Boot":
        order = np.argsort(patients, kind="mergesort")
        sorted_p = patients[order]
        bounds = np.flatnonzero(np.diff(sorted_p)) + 1
        groups = np.split(order, bounds)
        return cls(groups=groups, rng=np.random.default_rng(seed))

    @property
    def n_patients(self) -> int:
        return len(self.groups)

    def draws(self, n_boot: int = N_BOOT):
        g = self.groups
        m = len(g)
        for _ in range(n_boot):
            pick = self.rng.integers(0, m, size=m)
            yield np.concatenate([g[i] for i in pick])


def ci_table(fn, boot: Boot, keys, n_boot: int = N_BOOT) -> dict:
    acc = {k: [] for k in keys}
    for idx in boot.draws(n_boot):
        vals = fn(idx)
        for k in keys:
            v = vals.get(k, float("nan"))
            if v is not None and np.isfinite(v):
                acc[k].append(v)
    out = {}
    for k in keys:
        a = np.asarray(acc[k], dtype=float)
        if a.size >= 20:
            lo, hi = np.percentile(a, [2.5, 97.5])
            out[f"{k}_ci_low"] = float(lo)
            out[f"{k}_ci_high"] = float(hi)
            out[f"{k}_n_boot_used"] = int(a.size)
        else:
            out[f"{k}_ci_low"] = float("nan")
            out[f"{k}_ci_high"] = float("nan")
            out[f"{k}_n_boot_used"] = int(a.size)
    return out


def paired_delta(fn_a, fn_b, boot: Boot, keys, n_boot: int = N_BOOT) -> dict:
    """Paired patient bootstrap on identical rows: metric(a) - metric(b)."""
    base_a, base_b = fn_a(None), fn_b(None)
    acc = {k: [] for k in keys}
    for idx in boot.draws(n_boot):
        va, vb = fn_a(idx), fn_b(idx)
        for k in keys:
            d = va.get(k, np.nan) - vb.get(k, np.nan)
            if np.isfinite(d):
                acc[k].append(d)
    rows = []
    for k in keys:
        a = np.asarray(acc[k], dtype=float)
        lo, hi = (np.percentile(a, [2.5, 97.5]) if a.size >= 20 else (np.nan, np.nan))
        rows.append(
            {
                "metric": k,
                "value_a": base_a.get(k, np.nan),
                "value_b": base_b.get(k, np.nan),
                "delta": base_a.get(k, np.nan) - base_b.get(k, np.nan),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                "n_boot_used": int(a.size),
            }
        )
    return rows


# ------------------------------------------------------------ arm builders


@dataclass
class BinaryArm:
    group: str
    model: str
    protocol: str
    cohort: str
    source: str
    pos: np.ndarray
    score: np.ndarray
    called: np.ndarray
    patients: np.ndarray
    notes: str = ""

    def core(self, idx=None) -> dict:
        if idx is None:
            return binary_core(self.pos, self.score, self.called)
        return binary_core(self.pos[idx], self.score[idx], self.called[idx])

    def row(self, n_boot: int = N_BOOT) -> dict:
        boot = Boot.build(self.patients)
        r = {
            "group": self.group,
            "model": self.model,
            "task": "binary TD vs PIGD (positive = PIGD)",
            "evaluation_protocol": self.protocol,
            "cohort": self.cohort,
            "n_patients": boot.n_patients,
        }
        r.update(self.core())
        r["n_distinct_scores"] = int(np.unique(self.score).size)
        r["score_degenerate"] = bool(r["n_distinct_scores"] <= 2)
        r.update(ci_table(self.core, boot, BIN_CI, n_boot))
        r["source_path"] = self.source
        r["notes"] = self.notes
        return r


@dataclass
class ThreeClassArm:
    group: str
    model: str
    protocol: str
    cohort: str
    source: str
    y_idx: np.ndarray
    P: np.ndarray
    pred_idx: np.ndarray
    patients: np.ndarray
    notes: str = ""

    def core(self, idx=None) -> dict:
        if idx is None:
            return three_class_core(self.y_idx, self.P, self.pred_idx)
        return three_class_core(self.y_idx[idx], self.P[idx], self.pred_idx[idx])

    def row(self, n_boot: int = N_BOOT) -> dict:
        boot = Boot.build(self.patients)
        r = {
            "group": self.group,
            "model": self.model,
            "task": "three-class (TD / Indeterminate / PIGD)",
            "evaluation_protocol": self.protocol,
            "cohort": self.cohort,
            "n_patients": boot.n_patients,
        }
        r.update(self.core())
        r["score_degenerate_any_class"] = bool(
            min(r[f"n_distinct_scores_{k}"] for k in LABELS) <= 2
        )
        r.update(ci_table(self.core, boot, MC_CI, n_boot))
        r["source_path"] = self.source
        r["notes"] = self.notes
        return r


def lab_idx(s: pd.Series) -> np.ndarray:
    return s.map({k: i for i, k in enumerate(LABELS)}).to_numpy(dtype=int)


# ------------------------------------------------------------- loaders


def load_binary_from_long(
    frame: pd.DataFrame,
    *,
    group: str,
    protocol: str,
    cohort: str,
    source: str,
    score_col: str = "P_PIGD",
    pred_col: str = "predicted_state",
    label_col: str = "actual_next_label",
    mask: pd.Series | None = None,
    sort_cols: list[str] | None = None,
    notes: str = "",
) -> list[BinaryArm]:
    arms = []
    d = frame if mask is None else frame[mask]
    for model, g in d.groupby("model", sort=False):
        if sort_cols:
            g = g.sort_values(sort_cols, kind="mergesort")
        arms.append(
            BinaryArm(
                group=group,
                model=str(model),
                protocol=protocol,
                cohort=cohort,
                source=source,
                pos=(g[label_col] == "PIGD").to_numpy(),
                score=g[score_col].to_numpy(dtype=float),
                called=(g[pred_col] == "PIGD").to_numpy(),
                patients=g["PATNO"].to_numpy(),
                notes=notes,
            )
        )
    return arms


def load_three_from_long(
    frame: pd.DataFrame,
    *,
    group: str,
    protocol: str,
    cohort: str,
    source: str,
    prob_cols: tuple[str, str, str],
    label_col: str = "actual_next_label",
    pred_col: str | None = None,
    sort_cols: list[str] | None = None,
    notes: str = "",
) -> list[ThreeClassArm]:
    arms = []
    for model, g in frame.groupby("model", sort=False):
        if sort_cols:
            g = g.sort_values(sort_cols, kind="mergesort")
        P = g[list(prob_cols)].to_numpy(dtype=float)
        P = P / P.sum(axis=1, keepdims=True)
        if pred_col is None:
            pred = np.argmax(P, axis=1)
        else:
            pred = lab_idx(g[pred_col])
        arms.append(
            ThreeClassArm(
                group=group,
                model=str(model),
                protocol=protocol,
                cohort=cohort,
                source=source,
                y_idx=lab_idx(g[label_col]),
                P=P,
                pred_idx=pred,
                patients=g["PATNO"].to_numpy(),
                notes=notes,
            )
        )
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    cfg = ap.parse_args()
    out = Path(cfg.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nb = cfg.n_boot
    log: list[str] = []

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        log.append(msg)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # ---------------------------------------------------------- load
    say("Loading saved prediction files")
    prof = pd.read_csv(RUNS["profile"] / "derived_label_predictions.csv", low_memory=False)
    lopo = pd.read_csv(RUNS["lopo"] / "next_visit_predictions.csv", low_memory=False)
    band = pd.read_csv(RUNS["band"] / "next_visit_predictions.csv", low_memory=False)
    two = pd.read_csv(RUNS["two_state"] / "predictions.csv", low_memory=False)
    harm = pd.read_csv(RUNS["harmonised"] / "harmonised_predictions.csv")

    RAW = ("p_TD_raw", "p_Indeterminate_raw", "p_PIGD_raw")

    # ---------------------------------------------------------- validation
    say("\nValidation of the saved prediction files")
    for name, d, cols in [
        ("profile", prof, RAW),
        ("LOPO", lopo, RAW),
        ("band", band, RAW),
    ]:
        s = d[list(cols)].sum(axis=1)
        check(
            f"{name}: three-class probabilities sum to 1",
            bool(np.allclose(s, 1.0, atol=1e-9)),
            f"min {s.min():.12f} max {s.max():.12f}",
        )
        check(
            f"{name}: no missing probabilities",
            bool(d[list(cols)].notna().all().all()),
            f"{int(d[list(cols)].isna().sum().sum())} missing cells",
        )
    # the two-state run holds arms of two different output spaces in one file
    tcols = ["p_TD", "p_Indeterminate", "p_PIGD"]
    three_arm_mask = two["three_class_defined"].fillna(False).astype(bool)
    st = two.loc[three_arm_mask, tcols].sum(axis=1)
    check(
        "two-state: three-class probabilities sum to 1 on the three-class arms",
        bool(np.allclose(st, 1.0, atol=1e-9)),
        f"min {st.min():.12f} max {st.max():.12f}",
    )
    check(
        "two-state: no missing probabilities on the three-class arms",
        bool(two.loc[three_arm_mask, tcols].notna().all().all()),
        f"{int(two.loc[three_arm_mask, tcols].isna().sum().sum())} missing cells",
    )
    n_two_by_two = int(two["p_Indeterminate"].isna().sum())
    check(
        "two-state: missing Indeterminate probabilities are confined to the 2x2 arms",
        bool(two.loc[two["p_Indeterminate"].isna(), "model"].str.contains("2 observed").all()),
        f"{n_two_by_two} cells, all in the two 2x2 arms, which have no "
        "Indeterminate output by construction",
    )
    stb = two.loc[two["p_Indeterminate"].isna(), ["p_TD", "p_PIGD"]].sum(axis=1)
    check(
        "two-state: the 2x2 arms' two probabilities sum to 1",
        bool(np.allclose(stb, 1.0, atol=1e-9)),
        f"min {stb.min():.12f} max {stb.max():.12f}",
    )
    hm = sorted({c.split("|")[0] for c in harm.columns if "|" in c})
    hs = np.stack(
        [harm[[f"{m}|p_{k}" for k in LABELS]].to_numpy(dtype=float).sum(axis=1) for m in hm]
    )
    check(
        "harmonised: all 14 models' probabilities sum to 1",
        bool(np.allclose(hs, 1.0, atol=1e-9)),
        f"min {hs.min():.12f} max {hs.max():.12f}",
    )

    for name, d, keys in [
        ("profile", prof, KEY + ["model"]),
        ("LOPO", lopo, KEY + ["model"]),
        ("band", band, KEY + ["model"]),
    ]:
        dup = int(d.duplicated(subset=keys).sum())
        check(f"{name}: no duplicate predictions", dup == 0, f"{dup} duplicate key rows")
    dup_two = int(two.duplicated(subset=["PATNO", "target_event", "model"]).sum())
    check("two-state: no duplicate predictions", dup_two == 0, f"{dup_two} duplicate key rows")
    dup_h = int(harm.duplicated(subset=["patient", "n_history"]).sum())
    check("harmonised: no duplicate rows", dup_h == 0, f"{dup_h} duplicate key rows")

    for name, d, col in [
        ("profile", prof, "actual_next_label"),
        ("LOPO", lopo, "actual_next_label"),
        ("band", band, "actual_next_label"),
        ("two-state", two, "actual_label"),
        ("harmonised", harm, "actual"),
    ]:
        got = sorted(d[col].dropna().unique())
        check(f"{name}: label set", got == sorted(LABELS), f"{got}")

    # the two-class renormalisation convention
    sl = lopo[lopo["model"] == "CT-HMM (latent, interval-aware)"]
    den = sl["p_TD_raw"] + sl["p_PIGD_raw"]
    check(
        "LOPO: P_PIGD equals p_PIGD_raw renormalised over TD+PIGD",
        bool(np.allclose(sl["P_PIGD"], sl["p_PIGD_raw"] / den, atol=1e-9)),
        "confirms the two-class convention reused below for the harmonised run",
    )

    # cohort support counts across runs
    sup_lopo = lopo[lopo["model"] == lopo["model"].iloc[0]]["actual_next_label"].value_counts()
    sup_harm = harm["actual"].value_counts()
    check(
        "LOPO and harmonised share the same three-class support",
        bool(all(sup_lopo[k] == sup_harm[k] for k in LABELS)),
        ", ".join(f"{k}: {sup_lopo[k]} vs {sup_harm[k]}" for k in LABELS),
    )

    # ---------------------------------------------------------- arms
    say("\nBuilding evaluation arms")
    prof_bin = prof["eligible"] & prof["two_class_defined"]
    lopo_bin = lopo["eligible"] & lopo["two_class_defined"]
    band_bin = band["eligible"] & band["two_class_defined"]

    binary_arms: list[BinaryArm] = []
    three_arms: list[ThreeClassArm] = []

    # --- A: profile's own cohort
    binary_arms += load_binary_from_long(
        prof,
        group="A: rich profile, own cohort",
        protocol="5-fold patient CV x 4 repeats, repeat 0 reported",
        cohort="3,021 transitions with a defined TD/PIGD target",
        source=rel(RUNS["profile"] / "derived_label_predictions.csv"),
        mask=prof_bin,
        sort_cols=KEY,
    )
    three_arms += load_three_from_long(
        prof,
        group="A: rich profile, own cohort",
        protocol="5-fold patient CV x 4 repeats, repeat 0 reported",
        cohort="3,353 transitions",
        source=rel(RUNS["profile"] / "derived_label_predictions.csv"),
        prob_cols=RAW,
        sort_cols=KEY,
        notes="hard three-class prediction = argmax of the saved probabilities; "
        "this run reported only the two-class decision",
    )

    # --- B: LOPO subtype-history cohort
    binary_arms += load_binary_from_long(
        lopo,
        group="B: subtype-history models, LOPO",
        protocol="leave-one-patient-out, 934 folds",
        cohort="3,029 transitions with a defined TD/PIGD target",
        source=rel(RUNS["lopo"] / "next_visit_predictions.csv"),
        mask=lopo_bin,
        sort_cols=KEY,
    )
    three_arms += load_three_from_long(
        lopo,
        group="B: subtype-history models, LOPO",
        protocol="leave-one-patient-out, 934 folds",
        cohort="3,361 transitions",
        source=rel(RUNS["lopo"] / "next_visit_predictions.csv"),
        prob_cols=RAW,
        sort_cols=KEY,
        notes="hard three-class prediction = argmax of the saved probabilities",
    )

    # --- C: matched set, profile against CT-HMM
    pk = prof.loc[prof_bin, KEY].drop_duplicates()
    lk = lopo.loc[lopo_bin, KEY].drop_duplicates()
    matched = pk.merge(lk, on=KEY, how="inner")
    say(f"  matched set: {len(pk)} profile x {len(lk)} LOPO -> {len(matched)} shared transitions")
    mkey = set(map(tuple, matched.to_numpy()))

    def in_matched(d: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [tuple(r) in mkey for r in d[KEY].to_numpy()], index=d.index
        )

    binary_arms += load_binary_from_long(
        prof,
        group="C: matched comparison set",
        protocol="profile arm: 5-fold patient CV, repeat 0",
        cohort=f"{len(matched):,} transitions scored by both families",
        source=rel(RUNS["profile"] / "derived_label_predictions.csv"),
        mask=prof_bin & in_matched(prof),
        sort_cols=KEY,
        notes="restricted to the transitions both families score",
    )
    binary_arms += load_binary_from_long(
        lopo,
        group="C: matched comparison set",
        protocol="history arm: leave-one-patient-out",
        cohort=f"{len(matched):,} transitions scored by both families",
        source=rel(RUNS["lopo"] / "next_visit_predictions.csv"),
        mask=lopo_bin & in_matched(lopo),
        sort_cols=KEY,
        notes="restricted to the transitions both families score",
    )

    # --- D: ratio-band sweep
    binary_arms += load_binary_from_long(
        band,
        group="D: ratio-band sweep",
        protocol="leave-one-patient-out, bands refitted per fold",
        cohort="3,029 transitions with a defined TD/PIGD target",
        source=rel(RUNS["band"] / "next_visit_predictions.csv"),
        mask=band_bin,
        sort_cols=KEY,
    )
    three_arms += load_three_from_long(
        band,
        group="D: ratio-band sweep",
        protocol="leave-one-patient-out, bands refitted per fold",
        cohort="3,361 transitions",
        source=rel(RUNS["band"] / "next_visit_predictions.csv"),
        prob_cols=RAW,
        sort_cols=KEY,
        notes="hard three-class prediction = argmax of the saved probabilities",
    )

    # --- E: two-state family
    tkey = ["PATNO", "target_event"]
    n_arms = two["model"].nunique()
    present = two.groupby(tkey)["model"].nunique()
    complete = set(present[present == n_arms].index)
    staged = two[[tuple(r) in complete for r in two[tkey].to_numpy()]]
    ok = staged.groupby(tkey)["two_class_defined"].all() & staged.groupby(tkey)[
        "P_PIGD"
    ].apply(lambda s: s.notna().all())
    common_keys = set(ok[ok].index)
    tc = staged[[tuple(r) in common_keys for r in staged[tkey].to_numpy()]]
    say(
        f"  two-state: {len(complete):,} of {present.size:,} transitions are scored by "
        f"all {n_arms} arms; {len(common_keys):,} are scorable by all"
    )
    n_common = tc.groupby("model").size().unique()
    say(f"  two-state common two-class set: {n_common} rows per model")
    binary_arms += load_binary_from_long(
        tc,
        group="E: two-state HMM family",
        protocol="leave-one-patient-out",
        cohort=f"{int(n_common[0]):,} transitions scorable by every arm",
        source=rel(RUNS["two_state"] / "predictions.csv"),
        pred_col="predicted_two_class",
        label_col="actual_label",
        sort_cols=["PATNO", "target_event"],
        notes="common set across all six arms, as the source run defines it",
    )
    tt = two[two["three_class_defined"].fillna(False)]
    three_arms += load_three_from_long(
        tt,
        group="E: two-state HMM family",
        protocol="leave-one-patient-out",
        cohort="3,361 transitions",
        source=rel(RUNS["two_state"] / "predictions.csv"),
        prob_cols=("p_TD", "p_Indeterminate", "p_PIGD"),
        label_col="actual_label",
        pred_col="predicted_three_class",
        sort_cols=["PATNO", "target_event"],
        notes="three-class arms only; the 2x2 arms have no Indeterminate output",
    )

    # --- F: harmonised family (wide -> long)
    long_rows = []
    for m in hm:
        P = harm[[f"{m}|p_{k}" for k in LABELS]].to_numpy(dtype=float)
        long_rows.append(
            pd.DataFrame(
                {
                    "PATNO": harm["patient"].to_numpy(),
                    "n_history": harm["n_history"].to_numpy(),
                    "model": m,
                    "actual_next_label": harm["actual"].to_numpy(),
                    "p_TD_raw": P[:, 0],
                    "p_Indeterminate_raw": P[:, 1],
                    "p_PIGD_raw": P[:, 2],
                }
            )
        )
    hlong = pd.concat(long_rows, ignore_index=True)
    den = hlong["p_TD_raw"] + hlong["p_PIGD_raw"]
    hlong["P_PIGD"] = np.where(den > 0, hlong["p_PIGD_raw"] / den, np.nan)
    hlong["two_class_defined"] = den > 0
    hlong["eligible"] = hlong["actual_next_label"].isin(["TD", "PIGD"])
    hlong["predicted_state"] = np.where(hlong["P_PIGD"] >= 0.5, "PIGD", "TD")
    three_arms += load_three_from_long(
        hlong,
        group="F: harmonised subscore family",
        protocol="leave-one-patient-out, one protocol for all 14 models",
        cohort="3,361 transitions",
        source=rel(RUNS["harmonised"] / "harmonised_predictions.csv"),
        prob_cols=RAW,
        sort_cols=["PATNO", "n_history"],
        notes="hard three-class prediction = argmax of the saved probabilities",
    )
    hbin = hlong["eligible"] & hlong["two_class_defined"]
    # label persistence has no TD/PIGD probability when the previous label was
    # Indeterminate, so it is scorable on fewer cases than the other 13 models.
    hcount = hlong[hbin].groupby(["PATNO", "n_history"])["model"].nunique()
    hcommon = set(hcount[hcount == len(hm)].index)
    hbin_common = hbin & pd.Series(
        [tuple(r) in hcommon for r in hlong[["PATNO", "n_history"]].to_numpy()],
        index=hlong.index,
    )
    n_hc = int(hbin_common.sum() // len(hm))
    per_model = hlong[hbin].groupby("model").size()
    say(
        "  harmonised binary cohort sizes per model: "
        + ", ".join(f"{n:,} ({c} models)" for n, c in
                    sorted(per_model.value_counts().items(), reverse=True))
        + f"; {n_hc:,} cases are scorable by all {len(hm)}"
    )
    binary_arms += load_binary_from_long(
        hlong,
        group="F: harmonised subscore family",
        protocol="leave-one-patient-out, one protocol for all 14 models",
        cohort="each model on every case it can score (sizes differ, see notes)",
        source=rel(RUNS["harmonised"] / "harmonised_predictions.csv"),
        mask=hbin,
        sort_cols=["PATNO", "n_history"],
        notes="binary view derived here by renormalising the saved three-class "
        "probabilities over TD+PIGD, the same convention the LOPO run uses; "
        "the source run published three-class metrics only; cohort sizes differ "
        "between models, so use group F2 to rank them",
    )
    binary_arms += load_binary_from_long(
        hlong,
        group="F2: harmonised, common binary subset",
        protocol="leave-one-patient-out, one protocol for all 14 models",
        cohort=f"{n_hc:,} transitions scorable by all 14 models",
        source=rel(RUNS["harmonised"] / "harmonised_predictions.csv"),
        mask=hbin_common,
        sort_cols=["PATNO", "n_history"],
        notes="the comparable binary view of this family: every model scored on "
        "identical rows",
    )

    say(f"\nArms built: {len(binary_arms)} binary, {len(three_arms)} three-class")
    for label, arms in (("binary", binary_arms), ("three-class", three_arms)):
        by_group: dict[str, set[int]] = {}
        for a in arms:
            n = a.pos.size if isinstance(a, BinaryArm) else a.y_idx.size
            by_group.setdefault(a.group, set()).add(int(n))
        for g, sizes in by_group.items():
            check(
                f"{label} group '{g}' has one cohort size",
                len(sizes) == 1,
                f"sizes {sorted(sizes)}"
                + ("" if len(sizes) == 1 else " - rows in this group are NOT ranked together"),
            )

    # ---------------------------------------------------------- compute
    say("\nComputing binary metrics with patient bootstrap CIs")
    bin_rows = []
    for a in binary_arms:
        bin_rows.append(a.row(nb))
        say(f"  {a.group[:3]} {a.model[:52]:<52} acc {bin_rows[-1]['accuracy']:.4f} "
            f"auc {bin_rows[-1]['auc_roc']:.4f}")
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(out / "binary_comparison.csv", index=False)

    say("\nComputing three-class metrics with patient bootstrap CIs")
    mc_rows = []
    for a in three_arms:
        mc_rows.append(a.row(nb))
        say(f"  {a.group[:3]} {a.model[:52]:<52} acc {mc_rows[-1]['accuracy']:.4f} "
            f"indAUC {mc_rows[-1]['auc_ovr_Indeterminate']:.4f} "
            f"macroF1 {mc_rows[-1]['macro_f1']:.4f}")
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(out / "three_class_comparison.csv", index=False)

    # ----------------------------------------- cross-check against published
    say("\nCross-checking recovered values against the published tables")
    cc: list[dict] = []

    def crosscheck(label, pub, keycol, mapping, rec, group, model_map=None, tol=1e-4):
        for _, prow in pub.iterrows():
            pm = str(prow[keycol])
            rm = (model_map or {}).get(pm, pm)
            sel = rec[(rec["group"] == group) & (rec["model"] == rm)]
            if sel.empty:
                cc.append(
                    {
                        "source": label,
                        "model": pm,
                        "metric": "(arm)",
                        "published": np.nan,
                        "recovered": np.nan,
                        "abs_diff": np.nan,
                        "status": "NO RECOVERED ARM",
                        "note": "",
                    }
                )
                continue
            r = sel.iloc[0]
            for pcol, (rcol, scale, note) in mapping.items():
                if pcol not in prow.index or pd.isna(prow[pcol]):
                    continue
                pv = float(prow[pcol])
                rv = float(r[rcol]) * scale
                cc.append(
                    {
                        "source": label,
                        "model": pm,
                        "metric": pcol,
                        "published": pv,
                        "recovered": rv,
                        "abs_diff": abs(pv - rv),
                        "status": "MATCH" if abs(pv - rv) <= tol else "DIFFERS",
                        "note": note,
                    }
                )

    ACC = ("accuracy", 1.0, "")
    crosscheck(
        rel(RUNS["lopo"] / "model_comparison.csv"),
        pd.read_csv(RUNS["lopo"] / "model_comparison.csv"),
        "model",
        {
            "accuracy": ACC,
            "brier_two_class": ("brier_binary", 1.0, ""),
            "log_loss": ("log_loss", 1.0, "clip 1e-12, as in next_visit.py"),
            "n_scored": ("n_scored", 1.0, ""),
        },
        bin_df,
        "B: subtype-history models, LOPO",
    )
    crosscheck(
        rel(RUNS["band"] / "model_metrics.csv"),
        pd.read_csv(RUNS["band"] / "model_metrics.csv"),
        "model",
        {
            "accuracy": ACC,
            "brier_two_class": ("brier_binary", 1.0, ""),
            "log_loss": ("log_loss", 1.0, ""),
            "auc_roc": ("auc_roc", 1.0, "independently recomputed here"),
            "pr_auc": ("pr_auc", 1.0, "independently recomputed here"),
            "n_scored": ("n_scored", 1.0, ""),
        },
        bin_df,
        "D: ratio-band sweep",
    )
    crosscheck(
        rel(RUNS["two_state"] / "common_two_class.csv"),
        pd.read_csv(RUNS["two_state"] / "common_two_class.csv"),
        "model",
        {
            "accuracy": ACC,
            "brier": ("brier_binary", 2.0, "published value is the summed two-class Brier, i.e. 2x the single-outcome form"),
            "log_loss": ("log_loss", 1.0, ""),
            "recall_PIGD": ("sensitivity_PIGD", 1.0, ""),
            "precision_PIGD": ("precision_PIGD", 1.0, ""),
            "f1_PIGD": ("f1_PIGD", 1.0, ""),
            "balanced_accuracy": ("balanced_accuracy", 1.0, ""),
            "n_scored": ("n_scored", 1.0, ""),
        },
        bin_df,
        "E: two-state HMM family",
    )
    crosscheck(
        rel(RUNS["two_state"] / "native_three_class.csv"),
        pd.read_csv(RUNS["two_state"] / "native_three_class.csv"),
        "model",
        {
            "accuracy": ACC,
            "macro_f1": ("macro_f1", 1.0, ""),
            "balanced_accuracy": ("balanced_accuracy", 1.0, ""),
            "brier": ("brier_multiclass_summed", 1.0, "summed over three classes"),
            "log_loss": ("log_loss", 1.0, ""),
            "recall_TD": ("sensitivity_TD", 1.0, ""),
            "recall_Indeterminate": ("sensitivity_Indeterminate", 1.0, ""),
            "recall_PIGD": ("sensitivity_PIGD", 1.0, ""),
            "precision_PIGD": ("precision_PIGD", 1.0, ""),
            "f1_Indeterminate": ("f1_Indeterminate", 1.0, ""),
            "n_scored": ("n_scored", 1.0, ""),
        },
        mc_df,
        "E: two-state HMM family",
    )
    crosscheck(
        rel(RUNS["harmonised"] / "harmonised_metrics.csv"),
        pd.read_csv(RUNS["harmonised"] / "harmonised_metrics.csv"),
        "model",
        {
            "accuracy": ACC,
            "brier_multiclass": ("brier_multiclass_summed", 1.0, "summed over three classes"),
            "log_loss": ("log_loss", 1.0, ""),
            "macro_f1": ("macro_f1", 1.0, ""),
            "recall_TD": ("sensitivity_TD", 1.0, ""),
            "recall_Indeterminate": ("sensitivity_Indeterminate", 1.0, ""),
            "recall_PIGD": ("sensitivity_PIGD", 1.0, ""),
            "n_predicted_TD": ("n_predicted_TD", 1.0, "confirms argmax as the published decision rule"),
            "n_predicted_Indeterminate": ("n_predicted_Indeterminate", 1.0, ""),
            "n_predicted_PIGD": ("n_predicted_PIGD", 1.0, ""),
        },
        mc_df,
        "F: harmonised subscore family",
    )
    pub_auc = pd.read_csv(RUNS["auc"] / "auc_metrics.csv")
    crosscheck(
        rel(RUNS["auc"] / "auc_metrics.csv"),
        pub_auc,
        "model",
        {"auc": ("auc_roc", 1.0, "matched 3,020-case set"), "n": ("n_scored", 1.0, "")},
        bin_df,
        "C: matched comparison set",
        model_map={
            "CT-HMM": "CT-HMM (latent, interval-aware)",
            "Rich clinical profile": "Profile: level + rate + medication",
            "Persistence rule (reference)": "Persistence (last non-Indeterminate)",
        },
    )
    pub_ss = pd.read_csv(RUNS["ic"].parent / "20261111T133201Z_consolidated_results" / "two_class_sensitivity_specificity.csv")
    pub_ss["model_clean"] = pub_ss["model"].str.replace("^Subtype-history: ", "", regex=True)
    crosscheck(
        "experiments/20261111T133201Z_consolidated_results/two_class_sensitivity_specificity.csv",
        pub_ss,
        "model_clean",
        {
            "accuracy_recomputed": ACC,
            "sensitivity_PIGD": ("sensitivity_PIGD", 1.0, ""),
            "specificity_TD": ("specificity_TD", 1.0, ""),
            "ppv_PIGD": ("precision_PIGD", 1.0, ""),
            "npv_TD": ("npv_TD", 1.0, ""),
        },
        bin_df,
        "C: matched comparison set",
    )
    cc_df = pd.DataFrame(cc)
    cc_df.to_csv(out / "crosscheck_vs_published.csv", index=False)
    n_match = int((cc_df["status"] == "MATCH").sum())
    n_diff = int((cc_df["status"] == "DIFFERS").sum())
    n_none = int((cc_df["status"] == "NO RECOVERED ARM").sum())
    say(f"  {n_match} comparisons match, {n_diff} differ, {n_none} arms unmatched")
    if n_diff:
        say(cc_df[cc_df["status"] == "DIFFERS"].to_string(index=False))

    # ------------------------------------------------------ paired contrasts
    say("\nPaired patient-bootstrap contrasts on identical rows")
    bin_by = {(a.group, a.model): a for a in binary_arms}
    mc_by = {(a.group, a.model): a for a in three_arms}
    contrasts = []

    def pair(kind, ga, ma, gb, mb, keys, label):
        store = bin_by if kind == "binary" else mc_by
        a, b = store.get((ga, ma)), store.get((gb, mb))
        if a is None or b is None:
            say(f"  missing arm for {label}")
            return
        if a.patients.size != b.patients.size or not np.array_equal(a.patients, b.patients):
            say(f"  SKIPPED {label}: rows are not aligned")
            return
        boot = Boot.build(a.patients)
        for row in paired_delta(a.core, b.core, boot, keys, nb):
            row.update(
                {
                    "contrast": label,
                    "task": kind,
                    "model_a": ma,
                    "model_b": mb,
                    "group_a": ga,
                    "group_b": gb,
                    "n_paired": int(a.patients.size),
                    "n_patients": boot.n_patients,
                }
            )
            contrasts.append(row)
        say(f"  {label}: {len(keys)} metrics")

    BK = ["accuracy", "sensitivity_PIGD", "specificity_TD", "precision_PIGD", "f1_PIGD",
          "balanced_accuracy", "auc_roc", "pr_auc", "brier_binary", "log_loss"]
    MK = ["accuracy", "balanced_accuracy", "macro_f1", "auc_ovr_Indeterminate",
          "sensitivity_Indeterminate", "macro_auc_ovr", "brier_multiclass_summed",
          "log_loss", "mean_predictive_loglik"]
    C = "C: matched comparison set"
    D = "D: ratio-band sweep"
    Eg = "E: two-state HMM family"
    F = "F: harmonised subscore family"
    A = "A: rich profile, own cohort"
    CT = "CT-HMM (latent, interval-aware)"
    pair("binary", C, "Profile: level + rate + medication", C, CT, BK,
         "rich profile (level+rate+medication) - CT-HMM, matched set")
    pair("binary", C, "Profile: level + rate, no medication", C, CT, BK,
         "rich profile without levodopa - CT-HMM, matched set")
    pair("binary", C, "Profile: level only (random walk)", C, CT, BK,
         "rich profile (level only) - CT-HMM, matched set")
    pair("binary", A, "Profile: level + rate, no medication", A,
         "Profile: level + rate + medication", BK,
         "removing levodopa from the rich profile, profile cohort")
    for w in ("0.05", "0.10", "0.20", "0.30", "0.50"):
        pair("binary", D, f"CT-HMM (subtype x ratio band, width {w})", D, CT, BK,
             f"ratio bands at width {w} - unbanded CT-HMM")
    pair("binary", D, "CT-HMM (subtype x ratio band, width 0.30)", D,
         "CT-HMM (subtype x ratio band, width 0.10)", BK,
         "ratio bands at width 0.30 - width 0.10")
    pair("binary", Eg, "CT-HMM (2 hidden x 3 observed)", Eg,
         "CT-HMM (3 hidden x 3 observed, existing model)", BK,
         "two hidden states - three hidden states, common set")
    pair("binary", F, "Subscore GP, independent", F, "CT-HMM, 3 latent states", BK,
         "subscore GP (independent) - CT-HMM, harmonised binary view")
    F2 = "F2: harmonised, common binary subset"
    pair("binary", F2, "Subscore GP, independent", F2, "CT-HMM, 3 latent states", BK,
         "subscore GP (independent) - CT-HMM, common binary subset")
    pair("binary", F2, "Persistence (recorded label)", F2, "CT-HMM, 3 latent states", BK,
         "label persistence - CT-HMM, common binary subset")
    pair("three_class", F, "CT-HMM, 3 latent states", F, "Persistence (recorded label)", MK,
         "CT-HMM - label persistence, harmonised three-class")
    pair("three_class", F, "Subscore GP, independent", F, "CT-HMM, 3 latent states", MK,
         "subscore GP (independent) - CT-HMM, harmonised three-class")
    for w in ("0.10", "0.30"):
        pair("three_class", D, f"CT-HMM (subtype x ratio band, width {w})", D, CT, MK,
             f"ratio bands at width {w} - unbanded CT-HMM, three-class")
    pair("three_class", Eg, "CT-HMM (2 hidden x 3 observed)", Eg,
         "CT-HMM (3 hidden x 3 observed, existing model)", MK,
         "two hidden states - three hidden states, three-class")
    ct_df = pd.DataFrame(contrasts)
    cols = ["contrast", "task", "model_a", "model_b", "metric", "value_a", "value_b",
            "delta", "ci_low", "ci_high", "excludes_zero", "n_paired", "n_patients",
            "n_boot_used", "group_a", "group_b"]
    ct_df = ct_df[cols]
    ct_df.to_csv(out / "paired_contrasts.csv", index=False)
    say(f"  wrote {len(ct_df)} paired comparisons")

    # ------------------------------------------------------------- AIC / BIC
    say("\nConsolidating information criteria")
    ic_a = pd.read_csv(RUNS["ic"] / "aic_bic.csv")
    ic_b = pd.read_csv(RUNS["two_state"] / "aic_bic.csv")
    rows = []
    for _, r in ic_a.iterrows():
        rows.append(
            {
                "comparable_group": r["group"],
                "model": r["model"],
                "likelihood_definition": r["likelihood"],
                "dataset": "frozen panel, 934 patients",
                "loglik": r["loglik"],
                "n_params": int(r["k"]),
                "n_likelihood_terms": int(r["n_terms"]),
                "aic": r["aic"],
                "bic": r["bic"],
                "aic_recomputed": -2 * r["loglik"] + 2 * r["k"],
                "bic_recomputed": -2 * r["loglik"] + r["k"] * np.log(r["n_terms"]),
                "source_path": rel(RUNS["ic"] / "aic_bic.csv"),
            }
        )
    for _, r in ic_b.iterrows():
        rows.append(
            {
                "comparable_group": (
                    "D: label-sequence likelihood, all labelled visits"
                    if r["observation_set"] == "all labelled visits"
                    else "E: label-sequence likelihood, TD/PIGD visits only"
                ),
                "model": r["model"],
                "likelihood_definition": f"joint PMF over label sequences ({r['observation_set']})",
                "dataset": "frozen panel, 934 patients",
                "loglik": r["loglik"],
                "n_params": int(r["n_params"]),
                "n_likelihood_terms": int(r["n_observations"]),
                "aic": r["aic"],
                "bic": r["bic"],
                "aic_recomputed": -2 * r["loglik"] + 2 * r["n_params"],
                "bic_recomputed": -2 * r["loglik"] + r["n_params"] * np.log(r["n_observations"]),
                "source_path": rel(RUNS["two_state"] / "aic_bic.csv"),
            }
        )
    ic_df = pd.DataFrame(rows)
    ic_df["aic_matches_recomputed"] = (ic_df["aic"] - ic_df["aic_recomputed"]).abs() < 0.02
    ic_df["bic_matches_recomputed"] = (ic_df["bic"] - ic_df["bic_recomputed"]).abs() < 0.02
    ic_df["delta_aic_within_group"] = ic_df.groupby("comparable_group")["aic"].transform(
        lambda s: s - s.min()
    )
    ic_df["delta_bic_within_group"] = ic_df.groupby("comparable_group")["bic"].transform(
        lambda s: s - s.min()
    )
    ic_df.to_csv(out / "aic_bic_comparison.csv", index=False)
    check(
        "AIC reproduces -2logL + 2k for every tabulated model",
        bool(ic_df["aic_matches_recomputed"].all()),
        f"{int(ic_df['aic_matches_recomputed'].sum())}/{len(ic_df)} within 0.02",
    )
    check(
        "BIC reproduces -2logL + k log n for every tabulated model",
        bool(ic_df["bic_matches_recomputed"].all()),
        f"{int(ic_df['bic_matches_recomputed'].sum())}/{len(ic_df)} within 0.02",
    )

    # -------------------------------------------------- continuous endpoint
    say("\nConsolidating the continuous clinical endpoint")
    cont = pd.read_csv(RUNS["profile_cont"] / "model_comparison.csv")
    cont_rerun = pd.read_csv(RUNS["no_ledd"] / "model_comparison_recomputed.csv")
    cont["source_path"] = rel(RUNS["profile_cont"] / "model_comparison.csv")
    cont_rerun["source_path"] = rel(RUNS["no_ledd"] / "model_comparison_recomputed.csv")
    merged = cont.merge(
        cont_rerun, on="model", suffixes=("", "_recomputed"), how="outer", validate="1:1"
    )
    for col in ("loglik_per_domain_observation", "rmse_standardised", "mae_standardised", "coverage_95"):
        merged[f"{col}_agrees"] = (
            merged[col] - merged[f"{col}_recomputed"]
        ).abs() < 5e-4
    merged.to_csv(out / "rich_profile_comparison.csv", index=False)
    agree_cols = [c for c in merged.columns if c.endswith("_agrees")]
    check(
        "continuous-endpoint metrics reproduce across the two profile runs",
        bool(merged[agree_cols].all().all()),
        f"{int(merged[agree_cols].to_numpy().sum())}/{merged[agree_cols].size} cells agree",
    )

    with open(out / "driver.log", "w") as fh:
        fh.write("\n".join(log) + "\n")
    pd.DataFrame(checks).to_csv(out / "validation_checks.csv", index=False)
    json.dump(
        {
            "n_binary_arms": len(binary_arms),
            "n_three_class_arms": len(three_arms),
            "n_matched": int(len(matched)),
            "n_boot": nb,
            "seed": SEED,
            "checks_failed": int(sum(c["status"] == "FAIL" for c in checks)),
            "crosscheck_match": n_match,
            "crosscheck_differs": n_diff,
            "crosscheck_unmatched": n_none,
            "n_paired_contrasts": int(len(ct_df)),
        },
        open(out / "stage1_summary.json", "w"),
        indent=2,
    )
    say(f"\nWrote binary_comparison.csv and three_class_comparison.csv to {rel(out)}")


if __name__ == "__main__":
    main()
