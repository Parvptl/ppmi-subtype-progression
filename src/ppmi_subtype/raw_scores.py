"""Item-level reconstruction of the tremor and PIGD scores (Part C1).

Rebuilds, per patient-visit, exactly what `panel.build_subscores` computes
(same loaders, `select_medication_state`, `deduplicate_visits`, `clean_items`,
`compute_subscores`), but keeps the 16 constituent items, the reason a score
is missing, integer item sums, and alternative label constructions used only as
sensitivity analyses:

* Part III-only scores (10 tremor items, 3 PIGD items; same thresholds -- which
  were validated for the 11/5-item construction, so sensitivity only);
* one missing PIGD item tolerated (`max_missing_pigd_items = 1`; the subscore
  is then the mean of the observed items, i.e. the missing item is implicitly
  set to the mean of the patient's other PIGD items at that visit);
* the closed `inner` band (`band_boundaries = "inner"`).

Nothing here changes the project's phenotype definition: every label comes from
the unchanged `subtype.classify` with an explicitly modified copy of the
subtype configuration. See docs/cluster_phase/10_PART_C_RESEARCH_PLAN.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import loaders
from .config import Config, SubtypeConfig
from .subtype import classify, clean_items, compute_subscores, deduplicate_visits, \
    select_medication_state

KEY = ["PATNO", "EVENT_ID"]
PART2_ITEMS = ("NP2TRMR", "NP2WALK", "NP2FREZ")
GENES = ("LRRK2", "GBA", "SNCA", "PRKN")
PIGD_GROUPS = ("0", "1", "2", "3", "4-5", "6+")


# ---------------------------------------------------------------------------
#  Item tables
# ---------------------------------------------------------------------------
def part3_only(scfg: SubtypeConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Tremor and PIGD item lists with the Part II (questionnaire) items removed."""
    return (tuple(c for c in scfg.tremor_items if c not in PART2_ITEMS),
            tuple(c for c in scfg.pigd_items if c not in PART2_ITEMS))


def item_visit_table(part2: pd.DataFrame, part3: pd.DataFrame, scfg: SubtypeConfig) -> pd.DataFrame:
    """One row per (PATNO, EVENT_ID): raw items, as `panel.build_subscores` builds them.

    Same steps and order as `build_subscores`: OFF/untreated selection of Part
    III, outer merge with Part II on the visit key, then `deduplicate_visits`.
    Raw item values are kept (101 and out-of-range included) so that reasons
    for missingness can be counted. `has_part3` marks visits with a Part III
    row surviving the medication filter.
    """
    p3 = select_medication_state(part3, scfg)
    p2_items = [c for c in scfg.all_items if c in part2.columns]
    p3_items = [c for c in scfg.all_items if c in p3.columns]
    carry = [c for c in (scfg.pdstate_column, scfg.treatment_column, "NHY") if c in p3.columns]
    merged = part2[KEY + p2_items].merge(
        p3[KEY + p3_items + carry].assign(has_part3=True), on=KEY, how="outer")
    merged["has_part3"] = merged["has_part3"].eq(True)
    merged = deduplicate_visits(merged, scfg)
    return merged.reset_index(drop=True)


def missingness_reasons(table: pd.DataFrame, scfg: SubtypeConfig) -> pd.DataFrame:
    """Per-visit counts of missing / sentinel / out-of-range items, and one reason code.

    Reason codes (first that applies), for visits where a subscore is missing:
      no_part3            no Part III exam survived the OFF/untreated filter
      unrateable_101      at least one constituent item is a missing-value code (101)
      out_of_range        at least one item outside the valid range (not a sentinel)
      item_missing        at least one item absent
    `labelled` visits get reason "labelled".
    """
    lo, hi = scfg.valid_score_range
    codes = set(scfg.missing_value_codes)
    out = pd.DataFrame(index=table.index)
    for name, items in (("tremor", scfg.tremor_items), ("pigd", scfg.pigd_items)):
        raw = table[list(items)].apply(pd.to_numeric, errors="coerce")
        out[f"{name}_n_101"] = raw.isin(codes).sum(axis=1)
        out[f"{name}_n_out_of_range"] = (raw.notna() & ~raw.isin(codes)
                                         & ((raw < lo) | (raw > hi))).sum(axis=1)
        out[f"{name}_n_absent"] = raw.isna().sum(axis=1)
        out[f"{name}_n_invalid"] = clean_items(table, items, scfg).isna().sum(axis=1)
    sc = compute_subscores(table, scfg)
    labelled = sc["subtype"].notna()
    any101 = (out.tremor_n_101 + out.pigd_n_101) > 0
    anyoor = (out.tremor_n_out_of_range + out.pigd_n_out_of_range) > 0
    reason = np.select(
        [labelled, ~table["has_part3"], any101, anyoor],
        ["labelled", "no_part3", "unrateable_101", "out_of_range"], default="item_missing")
    out["reason"] = reason
    return out


def item_sums(table: pd.DataFrame, scfg: SubtypeConfig, tremor_items=None, pigd_items=None):
    """Integer item sums; NaN unless every listed item is a valid score."""
    out = []
    for items in (tuple(tremor_items or scfg.tremor_items), tuple(pigd_items or scfg.pigd_items)):
        c = clean_items(table, items, scfg)
        out.append(c.sum(axis=1).where(c.notna().all(axis=1)))
    return out[0], out[1]


def labels_under(table: pd.DataFrame, scfg: SubtypeConfig, **overrides) -> pd.DataFrame:
    """Subscores and labels with a modified copy of the subtype config."""
    return compute_subscores(table, dataclasses.replace(scfg, **overrides))


def part3_only_labels(table: pd.DataFrame, scfg: SubtypeConfig) -> pd.DataFrame:
    """Part III-only means and label (sensitivity construction)."""
    ti, pi = part3_only(scfg)
    sub = dataclasses.replace(scfg, tremor_items=ti, pigd_items=pi)
    t = clean_items(table, ti, scfg)
    p = clean_items(table, pi, scfg)
    tm = t.mean(axis=1).where(t.notna().all(axis=1))
    pm = p.mean(axis=1).where(p.notna().all(axis=1))
    lab = classify(tm, pm, sub)
    return pd.DataFrame({"tremor_p3": tm, "pigd_p3": pm, "subtype_p3": lab,
                         "subtype_name_p3": lab.map(scfg.label_names)}, index=table.index)


# ---------------------------------------------------------------------------
#  Label margin on the item-sum lattice
# ---------------------------------------------------------------------------
def label_grid(scfg: SubtypeConfig, n_tremor: int = 11, n_pigd: int = 5) -> np.ndarray:
    """Label code for every (tremor sum, PIGD sum) cell, via the unchanged rule."""
    T, P = np.meshgrid(np.arange(4 * n_tremor + 1), np.arange(4 * n_pigd + 1), indexing="ij")
    lab = classify(pd.Series(T.ravel() / n_tremor), pd.Series(P.ravel() / n_pigd), scfg)
    return lab.to_numpy().reshape(T.shape)


def margin_grid(grid: np.ndarray) -> np.ndarray:
    """Minimum L1 number of +/-1 item-point steps from each cell to a different label."""
    T, P = np.indices(grid.shape)
    t, p, g = T.ravel(), P.ravel(), grid.ravel()
    dist = np.abs(t[:, None] - t[None, :]) + np.abs(p[:, None] - p[None, :])
    dist = np.where(g[:, None] != g[None, :], dist, np.iinfo(np.int64).max)
    return dist.min(axis=1).reshape(grid.shape)


def label_margin(tsum, psum, margins: np.ndarray) -> np.ndarray:
    """Look up the margin for each visit; NaN where a sum is missing."""
    tsum, psum = np.asarray(tsum, float), np.asarray(psum, float)
    out = np.full(tsum.shape, np.nan)
    ok = np.isfinite(tsum) & np.isfinite(psum)
    out[ok] = margins[tsum[ok].astype(int), psum[ok].astype(int)]
    return out


# ---------------------------------------------------------------------------
#  Groupings
# ---------------------------------------------------------------------------
def pigd_group(psum) -> pd.Series:
    s = pd.Series(psum, dtype=float)
    return pd.Series(np.select([s == 0, s == 1, s == 2, s == 3, s <= 5, s >= 6],
                               list(PIGD_GROUPS), default=""), index=s.index).replace("", np.nan)


def quintile_edges(values) -> np.ndarray:
    """Quintile cut points of the given values (recorded in the report)."""
    v = np.asarray(values, float)
    return np.quantile(v[np.isfinite(v)], [0.2, 0.4, 0.6, 0.8])


def quintile_group(values, edges) -> pd.Series:
    """Q1..Q5 with right-closed bins (-inf, e1], (e1, e2], ... ; ties fall in the lower bin.

    With integer sums some edges coincide; empty bins are then simply absent.
    """
    v = pd.Series(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges, float), v.to_numpy(), side="left")
    out = pd.Series([f"Q{i + 1}" for i in idx], index=v.index)
    return out.where(v.notna())


# ---------------------------------------------------------------------------
#  Cohorts
# ---------------------------------------------------------------------------
def genetic_pd_mask(curated: pd.DataFrame) -> pd.Series:
    """PD cohort (COHORT == 1) participants with a genetic subgroup label.

    `subgroup` combines enrolment cohort and IU genetic consensus; combined
    labels such as 'LRRK2 + GBA' are kept. Prodromal carriers (COHORT 4) and
    non-genetic subgroups (Sporadic PD, Normosmic, ...) are excluded.
    """
    sub = curated["subgroup"].astype("string").fillna("")
    tokens = sub.str.split(r"\s*\+\s*", regex=True)
    has_gene = tokens.apply(lambda ts: any(t in GENES for t in ts))
    return (pd.to_numeric(curated["COHORT"], errors="coerce") == 1) & has_gene


def genetic_pd_panel(cfg: Config, subscores: pd.DataFrame) -> pd.DataFrame:
    """Visit panel for the genetic-PD sensitivity cohort.

    Same visit filters as `panel.filter_cohort` (excluded events, max follow-up)
    and the same time axis (`panel._add_time_axis`); only the participant
    selection differs. `panel.filter_cohort(extended=True)` is NOT used: it
    selects on `subgroup` alone and would include prodromal carriers.
    """
    from .panel import _add_time_axis  # reuse, unchanged
    cur = loaders.load_curated_cut(cfg)
    cur = cur[genetic_pd_mask(cur)]
    cur = cur[~cur["EVENT_ID"].isin(set(cfg.cohort["excluded_events"]))]
    cur = cur[cur["YEAR"] <= cfg.cohort["max_follow_up_years"]]
    pan = cur.merge(subscores, on=KEY, how="left", validate="one_to_one")
    return _add_time_axis(pan, cfg)


def off_on_pairs(part2: pd.DataFrame, part3: pd.DataFrame, scfg: SubtypeConfig) -> pd.DataFrame:
    """Same-visit OFF and ON Part III exams, aligned on PATNO, EVENT_ID and INFODT.

    Each exam is combined with the visit's (single) Part II row, so the three
    Part II items are identical in both; Part III-only labels are therefore
    reported as well. A visit with more than one row in the same state is not
    paired (none exist at audit time). Descriptive only -- not test-retest.
    """
    st = part3[scfg.pdstate_column].astype("string").str.strip().str.upper()
    p3 = part3.assign(_state=st)
    p3 = p3[p3["_state"].isin(["OFF", "ON"])]
    dup = p3.duplicated(KEY + ["_state"], keep=False)
    p3 = p3[~dup]
    p2_items = [c for c in scfg.all_items if c in part2.columns]
    p2 = part2[KEY + p2_items].drop_duplicates(KEY, keep=False)
    p3_items = [c for c in scfg.all_items if c in p3.columns]
    extra = [c for c in ("INFODT", "HRPOSTMED", "ONOFFORDER", "DBSYN", "EXAMTM",
                         scfg.treatment_column) if c in p3.columns]
    rows = []
    for state in ("OFF", "ON"):
        s = p3[p3["_state"] == state][KEY + p3_items + extra].merge(p2, on=KEY, how="left")
        sc = compute_subscores(s, scfg)
        p3o = part3_only_labels(s, scfg)
        t, p = item_sums(s, scfg)
        s = s[KEY + extra].assign(subtype_name=sc["subtype_name"], tremor_sum=t, pigd_sum=p,
                                  subtype_name_p3=p3o["subtype_name_p3"])
        rows.append(s.rename(columns={c: f"{c}_{state}" for c in s.columns if c not in KEY}))
    pairs = rows[0].merge(rows[1], on=KEY, how="inner")
    if "INFODT_OFF" in pairs:
        pairs = pairs[pairs["INFODT_OFF"] == pairs["INFODT_ON"]]
    return pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Patient-cluster bootstrap of count tables
# ---------------------------------------------------------------------------
def patient_bootstrap_sums(counts: np.ndarray, B: int, seed: int) -> np.ndarray:
    """Bootstrap sums of per-patient count vectors.

    counts: (n_patients, k). Each replicate draws n_patients patients with
    replacement (a drawn patient brings all of their counts) and returns the
    column sums: shape (B, k).
    """
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    out = np.empty((B, counts.shape[1]))
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        out[b] = w @ counts
    return out


def transition_table(frame: pd.DataFrame, group_col: str, groups, B: int = 4000,
                     seed: int = 42, max_interval: float = 1.5) -> pd.DataFrame:
    """P(group at next visit | group now) by interval class, with patient bootstrap.

    `frame` holds one row per visit with PATNO, years_from_baseline and the
    group column; pairs use consecutive visits at which the group is defined.
    Interval classes: <= max_interval and > max_interval years.
    """
    df = frame[frame[group_col].notna()].sort_values(["PATNO", "years_from_baseline"])
    g = df.groupby("PATNO", sort=False)
    pairs = pd.DataFrame({"PATNO": df["PATNO"], "a": df[group_col],
                          "b": g[group_col].shift(-1),
                          "dt": g["years_from_baseline"].shift(-1) - df["years_from_baseline"]})
    pairs = pairs[pairs["b"].notna()]
    pairs["iv"] = np.where(pairs["dt"] <= max_interval, f"<= {max_interval}", f"> {max_interval}")
    G = list(groups)
    gi = {x: i for i, x in enumerate(G)}
    rows = []
    for iv, sub in pairs.groupby("iv"):
        pats = np.sort(sub["PATNO"].unique())
        pidx = {p: i for i, p in enumerate(pats)}
        C = np.zeros((len(pats), len(G) * len(G)))
        np.add.at(C, (sub["PATNO"].map(pidx).to_numpy(),
                      sub["a"].map(gi).to_numpy() * len(G) + sub["b"].map(gi).to_numpy()), 1)
        tot = C.sum(axis=0).reshape(len(G), len(G))
        bs = patient_bootstrap_sums(C, B, seed).reshape(B, len(G), len(G))
        with np.errstate(invalid="ignore", divide="ignore"):
            pb = bs / bs.sum(axis=2, keepdims=True)
        for i, a in enumerate(G):
            n_from = tot[i].sum()
            for j, b in enumerate(G):
                p = tot[i, j] / n_from if n_from else np.nan
                lo, hi = (np.nanquantile(pb[:, i, j], [0.025, 0.975]) if n_from else (np.nan, np.nan))
                rows.append({"interval": iv, "from": a, "to": b, "n_pairs": int(tot[i, j]),
                             "n_from": int(n_from), "prob": p, "ci_lo": lo, "ci_hi": hi,
                             "n_patients_from": int((C.reshape(len(pats), len(G), len(G))[:, i, :]
                                                     .sum(axis=1) > 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Illustration only: seeded example patients
# ---------------------------------------------------------------------------
def example_patients(seq_frame: pd.DataFrame, n_total: int = 16, seed: int = 42):
    """Stratified seeded draw: baseline label x visit-count tertile.

    `seq_frame`: one row per patient with columns PATNO, baseline_label,
    n_visits (patients with >= 2 labelled visits). Allocation: strata sorted by
    size (largest first, PATNO-free tie-break by stratum name); 2 patients to
    each stratum in order until fewer than 2 remain to allocate, the remainder
    1 each. If a stratum has fewer patients than allocated, all are taken and
    the shortfall passes to the next stratum in the same order (logged).
    Returns (selected frame, log list, tertile edges).
    """
    rng = np.random.default_rng(seed)
    f = seq_frame.copy()
    edges = np.quantile(f["n_visits"], [1 / 3, 2 / 3])
    f["tertile"] = np.searchsorted(edges, f["n_visits"], side="left") + 1
    strata = (f.groupby(["baseline_label", "tertile"]).size().rename("size").reset_index()
              .sort_values(["size", "baseline_label", "tertile"], ascending=[False, True, True]))
    alloc = []
    left = n_total
    for i in range(len(strata)):
        remaining_strata = len(strata) - i
        a = 2 if left - 2 >= remaining_strata - 1 else 1
        a = min(a, left)
        alloc.append(a)
        left -= a
    strata["alloc"] = alloc
    log, chosen, carry = [], [], 0
    for _, s in strata.iterrows():
        want = int(s["alloc"]) + carry
        pool = np.sort(f[(f.baseline_label == s.baseline_label) &
                         (f.tertile == s.tertile)]["PATNO"].to_numpy())
        take = min(want, len(pool))
        if take < want:
            log.append(f"stratum {s.baseline_label}/T{s.tertile}: wanted {want}, only {len(pool)}")
        carry = want - take
        if take:
            chosen += list(rng.choice(pool, take, replace=False))
    if carry:
        log.append(f"{carry} example slots could not be filled")
    return f[f.PATNO.isin(chosen)].copy(), log, edges, strata


# ---------------------------------------------------------------------------
#  Per-visit analysis frame
# ---------------------------------------------------------------------------
def subscores_from_items(table: pd.DataFrame, scfg: SubtypeConfig) -> pd.DataFrame:
    """KEY + `compute_subscores`, i.e. what `panel.build_subscores` returns."""
    return pd.concat([table[KEY], compute_subscores(table, scfg)], axis=1)


def build_visit_frame(panel: pd.DataFrame, table: pd.DataFrame, scfg: SubtypeConfig) -> pd.DataFrame:
    """Panel visits joined to item-level reconstruction and sensitivity labels.

    Adds: tremor_sum, pigd_sum, missingness counts and reason, has_part3,
    PDSTATE (re-joined), Part III-only label, the one-missing-PIGD-item label,
    the inner-band label, label margin, and `recon_*` columns holding the
    reconstructed primary subscores and label for validation against the panel.
    Visits absent from the item table get reason "no_item_rows".
    """
    t, p = item_sums(table, scfg)
    reasons = missingness_reasons(table, scfg)
    prim = compute_subscores(table, scfg)
    p3 = part3_only_labels(table, scfg)
    miss1 = labels_under(table, scfg, max_missing_pigd_items=1)
    inner = labels_under(table, scfg, band_boundaries="inner")
    add = pd.concat([
        table[KEY + [c for c in ("has_part3", scfg.pdstate_column) if c in table.columns]],
        pd.DataFrame({"tremor_sum": t, "pigd_sum": p,
                      "recon_tremor_score": prim["tremor_score"],
                      "recon_pigd_score": prim["pigd_score"],
                      "recon_subtype_name": prim["subtype_name"],
                      "subtype_name_miss1": miss1["subtype_name"],
                      "subtype_name_inner": inner["subtype_name"]}, index=table.index),
        reasons, p3], axis=1)
    out = panel.merge(add, on=KEY, how="left", validate="one_to_one")
    out["reason"] = out["reason"].fillna("no_item_rows")
    out["has_part3"] = out["has_part3"].eq(True)
    out["label_margin"] = label_margin(out["tremor_sum"], out["pigd_sum"],
                                       margin_grid(label_grid(scfg)))
    return out


def validate_reconstruction(frame: pd.DataFrame) -> dict:
    """Recomputed subscores and labels must equal the frozen panel's exactly."""
    res = {}
    for a, b in (("tremor_score", "recon_tremor_score"), ("pigd_score", "recon_pigd_score")):
        x, y = frame[a].to_numpy(float), frame[b].to_numpy(float)
        same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=0, atol=1e-12)
        res[a] = int((~same).sum())
    la, lb = frame["subtype_name"].astype("string"), frame["recon_subtype_name"].astype("string")
    res["subtype_name"] = int((~((la.isna() & lb.isna()) | (la == lb).fillna(False))).sum())
    res["visits_checked"] = int(len(frame))
    res["labelled_visits"] = int(frame["subtype_name"].notna().sum())
    res["exact"] = all(v == 0 for k, v in res.items() if k in ("tremor_score", "pigd_score",
                                                             "subtype_name"))
    return res
