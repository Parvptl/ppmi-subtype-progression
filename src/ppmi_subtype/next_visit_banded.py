"""One LOPO fold of the ratio-banded CT-HMM, against the label CT-HMM baseline.

The two models are fitted inside the same fold function on purpose. They then
cannot differ in anything except the observation alphabet: same held-out
patient, same training patients, same prediction cases, same seed, same restart
scheme, same emission floor, same convergence accounting. Both arms are refitted
from the frozen panel with the current code, so the comparison does not rest on
reading an older run's numbers.

What is fitted on the training patients only
--------------------------------------------
Everything, including the band structure. The merged tail-band edges are a
*learned preprocessing decision*, so they are derived inside the fold from the
training patients' ratios and then applied to the held-out patient.
:func:`fold_alphabet` takes the held-out patient out before it looks at a single
ratio, and :func:`assert_bands_training_only` re-checks that afterwards rather
than trusting the construction.

The held-out patient's ratios can therefore fall outside every training band
only in one direction --- above the largest training ratio --- and the topmost
tail band is closed as ``[lo, inf)`` precisely so that case needs no decision
taken with knowledge of the held-out value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import hmm, hmm_banded, hmm_stability, loo, next_visit as nv, ratio_bands as rb

#: The baseline arm: the published next-visit CT-HMM, unchanged, refitted here.
MODEL_LABEL = nv.MODEL_CTHMM
#: The experimental arm.
MODEL_BANDED = "CT-HMM (subtype x ratio band)"
#: A control, not a competitor: the *square* model refitted through the joint
#: code path over a one-category-per-subtype alphabet, where the joint model is
#: mathematically the square model. It differs from the baseline arm only in
#: the code and the gradient method, so any gap between the two measures the
#: implementation rather than the bands. Reported separately and never part of
#: the primary comparison.
MODEL_CONTROL = "CT-HMM (subtype labels, joint code path)"
MODELS = (MODEL_LABEL, MODEL_BANDED, MODEL_CONTROL)


def band_width_model(width: float) -> str:
    """Model name for one configuration of the band-width experiment.

    The width goes in the name so predictions from several configurations can
    share one frame and be scored by the same functions as every other model.
    """
    return f"CT-HMM (subtype x ratio band, width {width:.2f})"


# ---------------------------------------------------------------------------
#  Per-visit ratios, aligned to the LOPO sequences
# ---------------------------------------------------------------------------
@dataclass
class BandedData:
    """`loo.LOOData`'s sequences with each visit's ratio alongside its label.

    ``sequences`` is one dict per patient with ``PATNO``, ``labels`` (subtype
    index, identical to :class:`loo.LOOData`'s), ``ratios`` (NaN where the
    ratio is not computable) and ``gaps``. The alignment between labels and
    ratios is asserted at construction, not assumed.
    """

    sequences: list[dict]
    subtype_names: list
    td_threshold: float
    pigd_threshold: float
    band_boundaries: str

    @property
    def patients(self) -> list:
        return [s["PATNO"] for s in self.sequences]

    def by_patient(self) -> dict:
        return {s["PATNO"]: s for s in self.sequences}


def prepare(panel: pd.DataFrame, data: loo.LOOData, cfg_subtype) -> BandedData:
    """Attach each visit's ratio to the sequences `loo.prepare` already built.

    The panel is filtered and sorted exactly as :func:`loo.prepare` does, and
    the recovered subtype labels are checked against the sequences element by
    element. If the two ever disagreed, every band would be attached to the
    wrong visit, so this is an assertion rather than a comment.
    """
    states = list(data.states)
    index = {s: k for k, s in enumerate(states)}
    keep = set(data.patients)

    labelled = panel[panel["subtype"].notna() & panel["PATNO"].isin(keep)].sort_values(
        ["PATNO", "years_from_baseline"])

    ratios = {p: g["ratio"].to_numpy(dtype=float)
              for p, g in labelled.groupby("PATNO", sort=False)}
    labels = {p: g["subtype"].map(index).to_numpy(dtype=int)
              for p, g in labelled.groupby("PATNO", sort=False)}

    sequences = []
    for seq in data.sequences:
        patno = seq["PATNO"]
        if not np.array_equal(seq["labels"], labels[patno]):
            raise ValueError(
                f"patient {patno}: the panel's label order does not match the "
                "sequence the LOPO data was built from")
        if len(ratios[patno]) != len(seq["labels"]):
            raise ValueError(f"patient {patno}: ratio and label counts differ")
        sequences.append({"PATNO": patno, "labels": seq["labels"],
                          "ratios": ratios[patno], "gaps": seq["gaps"]})

    return BandedData(
        sequences=sequences,
        subtype_names=list(nv.STATE_NAMES),
        td_threshold=float(cfg_subtype.td_threshold),
        pigd_threshold=float(cfg_subtype.pigd_threshold),
        band_boundaries=str(cfg_subtype.band_boundaries),
    )


# ---------------------------------------------------------------------------
#  Fold-specific band structure
# ---------------------------------------------------------------------------
def fold_alphabet(
    bdata: BandedData,
    held_out=None,
    min_observations: int = rb.MIN_TAIL_OBSERVATIONS,
    min_patients: int = rb.MIN_TAIL_PATIENTS,
    main_width: float = rb.MAIN_WIDTH,
) -> tuple[rb.BandScheme, rb.JointAlphabet, pd.DataFrame]:
    """Build the band scheme from every patient except `held_out`.

    `held_out=None` builds it from the whole cohort, which is used for the
    experiment's global diagnostics and for nothing that is scored.

    `main_width` sets the fixed main-range resolution. It is not data-dependent
    --- it is a configuration of the experiment, identical in every fold --- so
    varying it does not touch the leakage argument, which concerns the tail
    merge alone.
    """
    train = [s for s in bdata.sequences if s["PATNO"] != held_out]
    if not train:
        raise ValueError("no training patients left to build bands from")

    ratios = np.concatenate([s["ratios"] for s in train])
    patients = np.concatenate([np.full(len(s["ratios"]), s["PATNO"], dtype=object)
                               for s in train])

    tail_lows, tail_diag = rb.merge_tail(ratios, patients,
                                         min_observations=min_observations,
                                         min_patients=min_patients)
    scheme = rb.BandScheme(tuple(tail_lows), bdata.td_threshold,
                           bdata.pigd_threshold, bdata.band_boundaries,
                           main_width)
    alphabet = rb.build_alphabet(scheme, bdata.subtype_names)
    return scheme, alphabet, tail_diag


def assert_bands_training_only(bdata: BandedData, held_out, scheme: rb.BandScheme,
                               min_observations: int = rb.MIN_TAIL_OBSERVATIONS,
                               min_patients: int = rb.MIN_TAIL_PATIENTS,
                               main_width: float = rb.MAIN_WIDTH) -> None:
    """Check the fold's band edges are what the training patients alone produce.

    Rebuilds them from the training patients and compares. A band structure
    that had seen the held-out patient would differ whenever that patient's
    ratios were what pushed a candidate band over the minimum --- which is
    exactly the leak this has to rule out.
    """
    rebuilt, _, _ = fold_alphabet(bdata, held_out, min_observations, min_patients,
                                  main_width)
    if not np.array_equal(np.asarray(rebuilt.tail_lows), np.asarray(scheme.tail_lows)):
        raise loo.LeakageError(
            f"fold {held_out}: band edges {scheme.tail_lows} are not the ones the "
            f"training patients produce ({rebuilt.tail_lows})")
    if rebuilt.main_width != scheme.main_width:                    # pragma: no cover
        raise loo.LeakageError(
            f"fold {held_out}: main width {scheme.main_width} is not the "
            f"configured {rebuilt.main_width}")


def joint_sequences(bdata: BandedData, alphabet: rb.JointAlphabet,
                    patients=None) -> list[dict]:
    """Recode each patient's visits as joint (subtype, band) categories."""
    wanted = None if patients is None else set(patients)
    out = []
    for seq in bdata.sequences:
        if wanted is not None and seq["PATNO"] not in wanted:
            continue
        bands = alphabet.scheme.band_index(seq["ratios"])
        out.append({"PATNO": seq["PATNO"],
                    "labels": alphabet.code(seq["labels"], bands),
                    "gaps": seq["gaps"]})
    return out


# ---------------------------------------------------------------------------
#  One fold
# ---------------------------------------------------------------------------
def _diagnostics(records, train_loglik: float) -> dict:
    """Convergence block in the same shape :mod:`next_visit` writes."""
    best = max(records, key=lambda r: r.loglik)
    return {
        "train_loglik": float(train_loglik),
        "scipy_success": bool(best.success),
        "proj_grad": float(best.proj_grad),
        "termination": best.message,
        "converged": bool(best.success or best.proj_grad < nv.PROJ_GRAD_TOL),
        "n_iter": int(best.n_iter),
        "n_fev": int(best.n_fev),
    }


def run_fold(
    data: loo.LOOData,
    bdata: BandedData,
    cases: pd.DataFrame,
    held_out,
    seed: int,
    n_restarts: int,
    emission_floor: float | None,
    min_observations: int = rb.MIN_TAIL_OBSERVATIONS,
    min_patients: int = rb.MIN_TAIL_PATIENTS,
    maxfun: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit both arms on every other patient and predict this patient's visits.

    Returns ``(predictions, fold_diagnostics)``. The predictions frame carries
    the schema :mod:`next_visit`'s evaluation functions read, so both arms are
    scored by exactly the code that scored the published baseline.
    """
    train_sequences = [s for s in data.sequences if s["PATNO"] != held_out]
    train_pairs = data.pairs[data.pairs["PATNO"] != held_out]
    loo.assert_no_leakage(held_out, train_sequences, train_pairs)

    test_cases = cases[cases["PATNO"] == held_out].sort_values("visit_index")
    test_cases = test_cases.reset_index(drop=True)
    n = len(test_cases)
    test_sequence = next(s for s in data.sequences if s["PATNO"] == held_out)

    raw: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}

    # -- baseline arm: the published CT-HMM over the three recorded labels ---
    starts = nv.audited_starts(len(data.states), len(data.states), emission_floor,
                               seed, n_restarts)
    label_fit, label_records = hmm_stability.fit_multistart(
        train_sequences, data.states, n_hidden=len(data.states),
        emission_floor=emission_floor, starts=starts, state_names=nv.STATE_NAMES)
    label_preds = hmm.predictive_distributions(label_fit, [test_sequence])
    label_preds = label_preds.sort_values("step").reset_index(drop=True)
    _check_alignment(label_preds, test_cases, held_out, MODEL_LABEL)
    raw[MODEL_LABEL] = label_preds[nv.PROB_COLS].to_numpy(dtype=float)
    diagnostics[MODEL_LABEL] = _diagnostics(label_records, label_fit.loglik)

    # -- experimental arm: the same chain over joint (subtype, band) categories
    scheme, alphabet, tail_diag = fold_alphabet(bdata, held_out,
                                                min_observations, min_patients)
    assert_bands_training_only(bdata, held_out, scheme,
                               min_observations, min_patients)

    train_joint = joint_sequences(bdata, alphabet,
                                  [s["PATNO"] for s in train_sequences])
    if held_out in {s["PATNO"] for s in train_joint}:              # pragma: no cover
        raise loo.LeakageError(f"held-out patient {held_out} is in the joint training set")
    test_joint = joint_sequences(bdata, alphabet, [held_out])

    banded_fit, banded_records = hmm_banded.fit_banded(
        train_joint, alphabet, bdata.subtype_names, n_hidden=len(data.states),
        emission_floor=emission_floor, seed=seed, n_restarts=n_restarts,
        maxfun=maxfun)
    banded_preds = hmm_banded.predictive_distributions(banded_fit, test_joint)
    banded_preds = banded_preds.sort_values("step").reset_index(drop=True)
    _check_alignment(banded_preds, test_cases, held_out, MODEL_BANDED)
    if not np.allclose(banded_preds["p_joint_sum"], 1.0, atol=1e-9):
        raise RuntimeError(
            f"fold {held_out}: joint probabilities did not sum to one before "
            f"aggregation (worst {abs(banded_preds['p_joint_sum'] - 1).max():.3e})")
    raw[MODEL_BANDED] = banded_preds[nv.PROB_COLS].to_numpy(dtype=float)
    diagnostics[MODEL_BANDED] = _diagnostics(banded_records, banded_fit.loglik)

    # -- control arm: the square model through the joint code path ------------
    flat = rb.flat_alphabet(bdata.subtype_names)
    flat_train = [{"PATNO": s["PATNO"], "labels": s["labels"], "gaps": s["gaps"]}
                  for s in train_sequences]
    control_fit, control_records = hmm_banded.fit_banded(
        flat_train, flat, bdata.subtype_names, n_hidden=len(data.states),
        emission_floor=emission_floor, seed=seed, n_restarts=n_restarts)
    control_preds = hmm_banded.predictive_distributions(
        control_fit, [{"PATNO": held_out, "labels": test_sequence["labels"],
                       "gaps": test_sequence["gaps"]}])
    control_preds = control_preds.sort_values("step").reset_index(drop=True)
    _check_alignment(control_preds, test_cases, held_out, MODEL_CONTROL)
    raw[MODEL_CONTROL] = control_preds[nv.PROB_COLS].to_numpy(dtype=float)
    diagnostics[MODEL_CONTROL] = _diagnostics(control_records, control_fit.loglik)

    # -- the held-out patient's own visits, for the per-case record -----------
    held_seq = bdata.by_patient()[held_out]
    held_bands = scheme.band_index(held_seq["ratios"])
    held_codes = alphabet.code(held_seq["labels"], held_bands)
    k = test_cases["visit_index"].to_numpy(int)

    frames = []
    for name in MODELS:
        probs = np.asarray(raw[name], dtype=float)
        p_td, p_pigd, defined = nv.conditional_two_class(probs)
        frame = test_cases[[
            "PATNO", "visit_index", "n_history", "current_event", "current_years",
            "target_event", "target_years", "interval_years", "current_label",
            "actual_next_label", "eligible",
        ]].copy()
        frame["model"] = name
        frame["p_TD_raw"] = probs[:, nv.TD]
        frame["p_Indeterminate_raw"] = probs[:, nv.IND]
        frame["p_PIGD_raw"] = probs[:, nv.PIGD]
        frame["P_TD"] = p_td
        frame["P_PIGD"] = p_pigd
        frame["two_class_defined"] = defined
        frame["predicted_state"] = nv.predicted_state(p_td, p_pigd)
        frame["is_tie"] = defined & (p_td == p_pigd)
        frame["current_ratio"] = held_seq["ratios"][k]
        frame["current_category"] = [alphabet.labels[c] for c in held_codes[k]]
        frame["target_ratio"] = held_seq["ratios"][k + 1]
        for key, value in diagnostics[name].items():
            frame[f"fit_{key}"] = value
        frame["n_train_patients"] = len(train_sequences)
        frame["n_categories"] = alphabet.n_categories
        frame["n_tail_bands"] = len(scheme.tail_lows)
        frame["held_out_patient"] = held_out
        frames.append(frame)

    fold_diagnostics = {
        "held_out_patient": held_out,
        "n_train_patients": len(train_sequences),
        "n_cases": n,
        "n_tail_bands": len(scheme.tail_lows),
        "tail_lows": ",".join(f"{v:.2f}" for v in scheme.tail_lows),
        "tail_band_labels": "; ".join(
            scheme.band_label(b) for b in range(scheme.n_bands)
            if scheme.band_bounds(b)[0] >= rb.TAIL_START),
        "n_categories": alphabet.n_categories,
        "n_params_banded": banded_fit.n_params,
        "n_params_label": label_fit.n_params,
        "train_loglik_banded": float(banded_fit.loglik),
        "train_loglik_label": float(label_fit.loglik),
        "aic_banded": float(banded_fit.aic),
        "aic_label": float(label_fit.aic),
        "banded_min_diag_M": float(np.min(np.diag(banded_fit.M))),
        "label_min_diag_E": float(np.min(np.diag(label_fit.E))),
        "train_loglik_control": float(control_fit.loglik),
        "control_vs_label_loglik_gap": float(abs(control_fit.loglik - label_fit.loglik)),
        "control_vs_label_max_abs_E_gap": float(np.abs(control_fit.M - label_fit.E).max()),
        "control_vs_label_max_abs_prob_gap": float(np.abs(
            raw[MODEL_CONTROL] - raw[MODEL_LABEL]).max()) if len(test_cases) else 0.0,
        "held_out_max_ratio": float(np.nanmax(held_seq["ratios"]))
        if np.isfinite(held_seq["ratios"]).any() else float("nan"),
        "held_out_above_training_tail": bool(
            np.nanmax(held_seq["ratios"]) >= scheme.edges[-1]
            if np.isfinite(held_seq["ratios"]).any() else False),
        **{f"banded_{k}": v for k, v in diagnostics[MODEL_BANDED].items()
           if k in ("n_iter", "n_fev", "proj_grad", "converged", "termination")},
        **{f"label_{k}": v for k, v in diagnostics[MODEL_LABEL].items()
           if k in ("n_iter", "n_fev", "proj_grad", "converged", "termination")},
    }
    return pd.concat(frames, ignore_index=True), fold_diagnostics


def run_fold_widths(
    data: loo.LOOData,
    bdata: BandedData,
    cases: pd.DataFrame,
    held_out,
    seed: int,
    n_restarts: int,
    emission_floor: float | None,
    widths,
    min_observations: int = rb.MIN_TAIL_OBSERVATIONS,
    min_patients: int = rb.MIN_TAIL_PATIENTS,
    maxfun: int | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit the label baseline and one banded arm per main-range width.

    The fold of the band-width sensitivity experiment. Everything outside the
    alphabet is the fold :func:`run_fold` runs --- the same training split, the
    same leakage assertions, the same audited starts, the same prediction,
    alignment and normalisation checks --- so a width's arm differs from the
    published 0.1 arm in the band edges and in nothing else.

    The control arm is not refitted here. The previous experiment established
    that the joint code path reproduces the square model exactly (zero
    discordant cases), and repeating that once per width would cost a quarter
    of the compute to re-answer a settled question.

    Returns ``(predictions, fold_diagnostics)`` with one diagnostics row per
    arm, so the band structure of every width in every fold is recoverable.
    """
    widths = [float(w) for w in widths]
    train_sequences = [s for s in data.sequences if s["PATNO"] != held_out]
    train_pairs = data.pairs[data.pairs["PATNO"] != held_out]
    loo.assert_no_leakage(held_out, train_sequences, train_pairs)

    test_cases = cases[cases["PATNO"] == held_out].sort_values("visit_index")
    test_cases = test_cases.reset_index(drop=True)
    test_sequence = next(s for s in data.sequences if s["PATNO"] == held_out)

    raw: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    categories: dict[str, list[str]] = {}
    fold_rows: list[dict] = []

    held_seq = bdata.by_patient()[held_out]
    k = test_cases["visit_index"].to_numpy(int)
    held_max_ratio = (float(np.nanmax(held_seq["ratios"]))
                      if np.isfinite(held_seq["ratios"]).any() else float("nan"))

    # -- reference arm: the published CT-HMM over the three recorded labels --
    starts = nv.audited_starts(len(data.states), len(data.states), emission_floor,
                               seed, n_restarts)
    label_fit, label_records = hmm_stability.fit_multistart(
        train_sequences, data.states, n_hidden=len(data.states),
        emission_floor=emission_floor, starts=starts, state_names=nv.STATE_NAMES)
    label_preds = hmm.predictive_distributions(label_fit, [test_sequence])
    label_preds = label_preds.sort_values("step").reset_index(drop=True)
    _check_alignment(label_preds, test_cases, held_out, MODEL_LABEL)
    raw[MODEL_LABEL] = label_preds[nv.PROB_COLS].to_numpy(dtype=float)
    diagnostics[MODEL_LABEL] = _diagnostics(label_records, label_fit.loglik)
    categories[MODEL_LABEL] = [nv.STATE_NAMES[c] for c in held_seq["labels"][k]]
    fold_rows.append({
        "held_out_patient": held_out, "model": MODEL_LABEL, "main_width": float("nan"),
        "n_main_bands": 0, "n_tail_bands": 0, "tail_lows": "",
        "n_categories": len(data.states), "n_params": int(label_fit.n_params),
        "train_loglik": float(label_fit.loglik), "aic": float(label_fit.aic),
        "min_diag_emission": float(np.min(np.diag(label_fit.E))),
        "held_out_max_ratio": held_max_ratio,
        "held_out_above_training_tail": False,
        "n_train_patients": len(train_sequences), "n_cases": len(test_cases),
        **{key: diagnostics[MODEL_LABEL][key] for key in
           ("n_iter", "n_fev", "proj_grad", "converged", "termination")},
    })

    # -- one banded arm per configured main-range width ----------------------
    for width in widths:
        name = band_width_model(width)
        scheme, alphabet, _ = fold_alphabet(bdata, held_out, min_observations,
                                            min_patients, width)
        assert_bands_training_only(bdata, held_out, scheme, min_observations,
                                   min_patients, width)

        train_joint = joint_sequences(bdata, alphabet,
                                      [s["PATNO"] for s in train_sequences])
        if held_out in {s["PATNO"] for s in train_joint}:          # pragma: no cover
            raise loo.LeakageError(
                f"held-out patient {held_out} is in the joint training set")
        test_joint = joint_sequences(bdata, alphabet, [held_out])

        fit, records = hmm_banded.fit_banded(
            train_joint, alphabet, bdata.subtype_names, n_hidden=len(data.states),
            emission_floor=emission_floor, seed=seed, n_restarts=n_restarts,
            maxfun=maxfun)
        preds = hmm_banded.predictive_distributions(fit, test_joint)
        preds = preds.sort_values("step").reset_index(drop=True)
        _check_alignment(preds, test_cases, held_out, name)
        if not np.allclose(preds["p_joint_sum"], 1.0, atol=1e-9):
            raise RuntimeError(
                f"fold {held_out}, width {width}: joint probabilities did not sum "
                f"to one before aggregation "
                f"(worst {abs(preds['p_joint_sum'] - 1).max():.3e})")
        raw[name] = preds[nv.PROB_COLS].to_numpy(dtype=float)
        diagnostics[name] = _diagnostics(records, fit.loglik)

        held_codes = alphabet.code(held_seq["labels"],
                                   scheme.band_index(held_seq["ratios"]))
        categories[name] = [alphabet.labels[c] for c in held_codes[k]]
        fold_rows.append({
            "held_out_patient": held_out, "model": name, "main_width": width,
            "n_main_bands": int(scheme.n_main_bands),
            "n_tail_bands": len(scheme.tail_lows),
            "tail_lows": ",".join(f"{v:.2f}" for v in scheme.tail_lows),
            "n_categories": int(alphabet.n_categories),
            "n_params": int(fit.n_params),
            "train_loglik": float(fit.loglik), "aic": float(fit.aic),
            "min_diag_emission": float(np.min(np.diag(fit.M))),
            "held_out_max_ratio": held_max_ratio,
            "held_out_above_training_tail": bool(
                held_max_ratio >= scheme.edges[-1]
                if np.isfinite(held_max_ratio) else False),
            "n_train_patients": len(train_sequences), "n_cases": len(test_cases),
            **{key: diagnostics[name][key] for key in
               ("n_iter", "n_fev", "proj_grad", "converged", "termination")},
        })

    # -- the per-case record, one block per arm ------------------------------
    frames = []
    for name in [MODEL_LABEL] + [band_width_model(w) for w in widths]:
        probs = np.asarray(raw[name], dtype=float)
        p_td, p_pigd, defined = nv.conditional_two_class(probs)
        frame = test_cases[[
            "PATNO", "visit_index", "n_history", "current_event", "current_years",
            "target_event", "target_years", "interval_years", "current_label",
            "actual_next_label", "eligible",
        ]].copy()
        frame["model"] = name
        frame["main_width"] = (float("nan") if name == MODEL_LABEL
                               else float(name.split("width ")[1].rstrip(")")))
        frame["p_TD_raw"] = probs[:, nv.TD]
        frame["p_Indeterminate_raw"] = probs[:, nv.IND]
        frame["p_PIGD_raw"] = probs[:, nv.PIGD]
        frame["P_TD"] = p_td
        frame["P_PIGD"] = p_pigd
        frame["two_class_defined"] = defined
        frame["predicted_state"] = nv.predicted_state(p_td, p_pigd)
        frame["is_tie"] = defined & (p_td == p_pigd)
        frame["current_ratio"] = held_seq["ratios"][k]
        frame["current_category"] = categories[name]
        frame["target_ratio"] = held_seq["ratios"][k + 1]
        for key, value in diagnostics[name].items():
            frame[f"fit_{key}"] = value
        frame["n_train_patients"] = len(train_sequences)
        frame["held_out_patient"] = held_out
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), fold_rows


def _check_alignment(preds: pd.DataFrame, test_cases: pd.DataFrame, held_out,
                     model: str) -> None:
    if len(preds) != len(test_cases) or not np.array_equal(
            preds["step"].to_numpy(int), test_cases["visit_index"].to_numpy(int)):
        raise RuntimeError(f"{model} predictions misaligned for patient {held_out}")


# ---------------------------------------------------------------------------
#  Band diagnostics, for the report
# ---------------------------------------------------------------------------
def band_occupancy(bdata: BandedData, alphabet: rb.JointAlphabet,
                   patients=None) -> pd.DataFrame:
    """Observations and distinct patients per joint category."""
    wanted = None if patients is None else set(patients)
    rows = []
    for seq in bdata.sequences:
        if wanted is not None and seq["PATNO"] not in wanted:
            continue
        bands = alphabet.scheme.band_index(seq["ratios"])
        codes = alphabet.code(seq["labels"], bands)
        for c, r in zip(codes.tolist(), seq["ratios"].tolist()):
            rows.append({"PATNO": seq["PATNO"], "category": c, "ratio": r})

    seen = pd.DataFrame(rows)
    counts = (seen.groupby("category")
              .agg(n_observations=("PATNO", "size"), n_patients=("PATNO", "nunique"))
              if len(seen) else pd.DataFrame(columns=["n_observations", "n_patients"]))

    out = pd.DataFrame({
        "category_index": range(alphabet.n_categories),
        "category": alphabet.labels,
        "subtype": [alphabet.subtype_names[s] for s, _ in alphabet.pairs],
        "band": [rb.UNDEFINED_BAND if b < 0 else alphabet.scheme.band_label(b)
                 for _, b in alphabet.pairs],
        "band_low": [np.nan if b < 0 else alphabet.scheme.band_bounds(b)[0]
                     for _, b in alphabet.pairs],
        "band_high": [np.nan if b < 0 else alphabet.scheme.band_bounds(b)[1]
                      for _, b in alphabet.pairs],
        "is_tail": [False if b < 0 else alphabet.scheme.band_bounds(b)[0] >= rb.TAIL_START
                    for _, b in alphabet.pairs],
    })
    out = out.join(counts, on="category_index").fillna({"n_observations": 0,
                                                        "n_patients": 0})
    out["n_observations"] = out["n_observations"].astype(int)
    out["n_patients"] = out["n_patients"].astype(int)
    return out


def ratio_range_accuracy(predictions: pd.DataFrame,
                         edges=(0.0, 0.9, 1.15, 2.0, 3.0, np.inf)) -> pd.DataFrame:
    """Accuracy per model by the ratio recorded at the prediction point.

    The breakpoints are the two classification thresholds plus 2.0 and 3.0 (the
    main-range/tail boundary), so the cut lines up with the band structure
    rather than with the quantiles of whatever the result turned out to be.
    Visits whose ratio is not computable are reported as their own group.
    """
    rows = []
    for name, g in predictions.groupby("model", sort=False):
        s = nv.scored(g)
        if not len(s):
            continue
        r = s["current_ratio"].to_numpy(dtype=float)
        groups = [("undefined", ~np.isfinite(r))]
        for lo, hi in zip(edges[:-1], edges[1:]):
            label = f"[{lo:.2f},inf)" if np.isinf(hi) else f"[{lo:.2f},{hi:.2f})"
            groups.append((label, np.isfinite(r) & (r >= lo) & (r < hi)))
        for label, mask in groups:
            sub = s[mask]
            rows.append({
                "model": name, "ratio_range": label, "n": int(len(sub)),
                "n_patients": int(sub["PATNO"].nunique()) if len(sub) else 0,
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan"),
                "brier_two_class": float(np.mean(
                    (sub["P_PIGD"] - (sub["actual_next_label"] == "PIGD")) ** 2))
                if len(sub) else float("nan"),
            })
    return pd.DataFrame(rows)
