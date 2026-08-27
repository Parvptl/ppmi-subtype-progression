"""Progression prediction for a patient seen for the first time.

Why this model exists
---------------------
The Markov and growth models in :mod:`ppmi_subtype.hmm` and
:mod:`ppmi_subtype.hbayes` earn their accuracy by filtering a patient's
*history* forward. A patient walking into clinic for the first time has no
history, and measured on that task the difference between those models and a
plain lookup table is +0.005 log-likelihood per prediction --- nothing.

What a new patient does have is a rich baseline assessment. Held out over four
independent splits, adding the baseline ratio and baseline clinical measures to
the subtype and horizon improves prediction by +0.022 log-likelihood per
prediction and lifts accuracy from 0.681 to 0.692, beating every longitudinal
model on this task.

So for the first-visit question the right model is not a longitudinal one at
all. It is a direct model of

    P(subtype at horizon h | everything known at the first visit)

fitted by multinomial logistic regression, which keeps the coefficients
readable --- a clinician can see which measures move the prediction.

Scope
-----
This model answers the first-visit question only. Once a patient has follow-up,
the hidden Markov model overtakes it, and by six visits its advantage over a
lookup table has grown to +0.047. The two are complementary and the choice is
determined by how much history exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STATE_NAMES = ["TD", "Indeterminate", "PIGD"]

#: Baseline measures available at a first visit. All are 97-100% complete in
#: the sporadic PD cohort. `rem`, `scopa`, `upsit` and `moca` are the
#: diffuse-synucleinopathy markers the proposal's second half is built on.
DEFAULT_COVARIATES = [
    "age",
    "SEX",
    "duration_yrs",
    "rem",                 # RBD screening questionnaire
    "scopa",               # SCOPA-AUT, autonomic dysfunction
    "upsit",               # smell identification
    "moca",                # cognition
    "MIA_PUTAMEN_BILAT",   # DaTscan putamen binding
    "hy",                  # Hoehn & Yahr
    "updrs2_score",        # MDS-UPDRS Part II total
]

RATIO_CLIP = 8.0


# ---------------------------------------------------------------------------
#  Training data
# ---------------------------------------------------------------------------
def build_training_frame(
    panel: pd.DataFrame,
    covariates: list[str] | None = None,
    label_col: str = "subtype",
    time_col: str = "years_from_baseline",
) -> pd.DataFrame:
    """One row per (patient, later visit): baseline features and the outcome.

    Every row shares the same predictors --- the patient's first labelled visit
    --- and differs only in the horizon and the observed subtype at that
    horizon. That is exactly the first-visit question, asked once per follow-up
    visit the patient contributed.
    """
    covariates = covariates or DEFAULT_COVARIATES
    df = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])
    first = df.groupby("PATNO").head(1).set_index("PATNO")

    rows = []
    for patno, group in df.groupby("PATNO", sort=False):
        labels = group[label_col].to_numpy()
        times = group[time_col].to_numpy()
        if len(labels) < 2:
            continue
        base = first.loc[patno]
        base_ratio = base.get("ratio", np.nan)
        for k in range(1, len(labels)):
            row = {
                "PATNO": patno,
                "outcome": int(labels[k] - 1),
                "baseline_state": int(labels[0] - 1),
                "horizon": float(times[k] - times[0]),
                "baseline_ratio": float(base_ratio) if np.isfinite(base_ratio) else np.nan,
            }
            row.update({c: base.get(c, np.nan) for c in covariates})
            rows.append(row)
    return pd.DataFrame(rows)


def design_matrix(
    frame: pd.DataFrame,
    covariates: list[str],
    use_ratio: bool = True,
    use_covariates: bool = True,
) -> pd.DataFrame:
    """Baseline state, horizon, their interaction, and optional baseline measures.

    The state-by-horizon interactions matter: a patient starting PIGD and one
    starting TD do not drift at the same rate, so the effect of time has to be
    allowed to differ by starting state.
    """
    states = pd.get_dummies(frame["baseline_state"], prefix="state")
    states = states.reindex(columns=["state_0", "state_1", "state_2"], fill_value=0)
    states = states.astype(float).reset_index(drop=True)

    h = frame["horizon"].to_numpy(dtype=float)
    parts = [states, pd.DataFrame({"horizon": h, "log_horizon": np.log1p(h)})]
    parts.append(
        pd.DataFrame({f"{c}_x_horizon": states[c].to_numpy() * h for c in states.columns})
    )
    if use_ratio:
        parts.append(
            pd.DataFrame({"baseline_ratio": frame["baseline_ratio"].clip(0, RATIO_CLIP).to_numpy()})
        )
    if use_covariates:
        parts.append(frame[covariates].astype(float).reset_index(drop=True))

    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------
@dataclass
class NewPatientModel:
    """Multinomial logistic model of subtype at a horizon, from a first visit."""

    covariates: list[str] = field(default_factory=lambda: list(DEFAULT_COVARIATES))
    use_ratio: bool = True
    use_covariates: bool = True
    C: float = 1.0
    seed: int = 42

    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    model: LogisticRegression | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "NewPatientModel":
        X = design_matrix(frame, self.covariates, self.use_ratio, self.use_covariates)
        self.feature_names = list(X.columns)

        # Imputer and scaler are fitted on training rows only, so no held-out
        # information reaches the fit.
        self.imputer = SimpleImputer(strategy="median").fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        self.model = LogisticRegression(
            max_iter=5000, C=self.C, random_state=self.seed
        ).fit(self.scaler.transform(self.imputer.transform(X)), frame["outcome"])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X = design_matrix(frame, self.covariates, self.use_ratio, self.use_covariates)
        X = X.reindex(columns=self.feature_names, fill_value=np.nan)
        P = self.model.predict_proba(self.scaler.transform(self.imputer.transform(X)))
        P = np.clip(P, 1e-6, None)
        return P / P.sum(axis=1, keepdims=True)

    # -------------------------------------------------------------- clinical
    def predict_patient(
        self, baseline: dict, horizons=(1, 2, 3, 5)
    ) -> pd.DataFrame:
        """The clinical output for one new patient.

        `baseline` needs ``baseline_state`` (0 = TD, 1 = Indeterminate,
        2 = PIGD) and whichever covariates are known; anything missing is
        imputed with the training median.
        """
        rows = []
        for h in horizons:
            row = {"horizon": float(h), "baseline_state": int(baseline["baseline_state"]),
                   "baseline_ratio": baseline.get("baseline_ratio", np.nan)}
            row.update({c: baseline.get(c, np.nan) for c in self.covariates})
            rows.append(row)
        frame = pd.DataFrame(rows)

        P = self.predict_proba(frame)
        out = pd.DataFrame(P, columns=STATE_NAMES)
        out.insert(0, "horizon_years", [float(h) for h in horizons])
        now = {s: 0.0 for s in STATE_NAMES}
        now[STATE_NAMES[int(baseline["baseline_state"])]] = 1.0
        return pd.concat(
            [pd.DataFrame([{"horizon_years": 0.0, **now}]), out], ignore_index=True
        )


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: NewPatientModel, frame: pd.DataFrame, label: str) -> dict:
    P = model.predict_proba(frame)
    j = frame["outcome"].to_numpy(int)
    p_true = P[np.arange(len(j)), j]
    onehot = np.zeros_like(P)
    onehot[np.arange(len(j)), j] = 1.0
    return {
        "model": label,
        "n": len(j),
        "loglik_per_prediction": float(np.log(p_true).mean()),
        "accuracy": float((P.argmax(axis=1) == j).mean()),
        "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
    }


def covariate_importance(
    train: pd.DataFrame, test: pd.DataFrame, covariates: list[str], seed: int = 42
) -> pd.DataFrame:
    """Held-out cost of dropping each covariate, one at a time.

    Reported as a drop in log-likelihood rather than as a coefficient, because
    the covariates are correlated and a coefficient in a multinomial model is
    not readable on its own.
    """
    full = NewPatientModel(covariates=covariates, seed=seed).fit(train)
    reference = evaluate(full, test, "full")["loglik_per_prediction"]

    rows = []
    for drop in covariates:
        reduced = [c for c in covariates if c != drop]
        m = NewPatientModel(covariates=reduced, seed=seed).fit(train)
        score = evaluate(m, test, drop)["loglik_per_prediction"]
        rows.append({"covariate": drop, "loglik_without": score,
                     "cost_of_dropping": reference - score})
    return pd.DataFrame(rows).sort_values("cost_of_dropping", ascending=False)


def calibration_table(
    model: NewPatientModel, frame: pd.DataFrame, target: int = 2, n_bins: int = 5
) -> pd.DataFrame:
    """Predicted versus observed rate for one outcome class, on held-out rows."""
    P = model.predict_proba(frame)[:, target]
    actual = (frame["outcome"].to_numpy(int) == target).astype(float)
    edges = np.unique(np.quantile(P, np.linspace(0, 1, n_bins + 1)))
    binned = pd.cut(P, bins=edges, include_lowest=True)
    return (
        pd.DataFrame({"bin": binned, "predicted": P, "observed": actual})
        .groupby("bin", observed=True)
        .agg(n=("observed", "size"), mean_predicted=("predicted", "mean"),
             observed_rate=("observed", "mean"))
        .reset_index()
    )
