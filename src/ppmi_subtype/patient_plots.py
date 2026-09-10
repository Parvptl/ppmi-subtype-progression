"""Patient-level views of the subtype data.

These figures answer questions that the population-level figures in
:mod:`ppmi_subtype.plots` cannot:

    * when a visit is labelled Indeterminate, where inside the band does the
      ratio actually sit?
    * what does one patient's course look like, rather than the cohort average?
    * how many patients move *towards* TD, not away from it?
    * what does the raw ratio do before any model is fitted?

Palette and styling are inherited from :mod:`ppmi_subtype.plots` so that the
whole report reads as one system.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .plots import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    STATE_COLORS,
    STATE_ORDER,
    SURFACE,
    _finish,
    _style,
)

FAINT = "#c9c8c4"

#: Bucket for a visit that has no preceding labelled visit to have come from.
NO_PREVIOUS = "first visit, no previous"


# ---------------------------------------------------------------------------
#  1. Where inside the Indeterminate band do these visits actually sit?
# ---------------------------------------------------------------------------
def indeterminate_boundary(
    values: pd.DataFrame,
    next_state: pd.DataFrame,
    path: Path,
    td_threshold: float = 1.15,
    pigd_threshold: float = 0.90,
) -> Path:
    """Distribution of the ratio at Indeterminate visits, and what happens next.

    `values` has one row per achievable ratio value inside the band and one
    column per previous state, holding counts. `next_state` has one row per
    ratio value group and columns ``TD``/``Indeterminate``/``PIGD`` holding the
    share of patients recorded in each state at the following visit.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), width_ratios=[3, 2])
    _style(ax)
    _style(ax2)

    # ---- left: the achievable values, stacked by where the patient came from
    ratios = values.index.to_numpy(dtype=float)
    y = np.arange(len(ratios))
    # A patient's first labelled visit has no previous state; it gets its own
    # neutral slot rather than being folded into one of the three subtypes.
    palette = dict(STATE_COLORS)
    palette[NO_PREVIOUS] = FAINT
    sources = [c for c in ("TD", "Indeterminate", "PIGD", NO_PREVIOUS)
               if c in values.columns]
    left = np.zeros(len(ratios))
    for src in sources:
        w = values[src].to_numpy(dtype=float)
        ax.barh(
            y, w, left=left, height=0.72,
            color=palette[src], edgecolor=SURFACE, linewidth=1.4,
            label=src if src == NO_PREVIOUS else f"came from {src}",
        )
        left += w

    total = values[sources].sum(axis=1).to_numpy(dtype=float)
    n_total = int(total.sum())
    for k, (val, n) in enumerate(zip(ratios, total)):
        ax.text(
            n + max(total) * 0.012, k, f"{int(n)}   ({n / total.sum():.0%})",
            va="center", ha="left", fontsize=9.5, color=INK,
            fontweight="bold" if n / total.sum() > 0.10 else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{v:.4f}" for v in ratios], fontsize=9.5)
    ax.set_xlim(0, max(total) * 1.28)
    ax.set_ylim(-0.8, len(ratios) - 0.2)
    ax.set_xlabel("number of visits", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("TD/PIGD ratio recorded at the visit", color=INK_MUTED, fontsize=10)
    ax.set_title(
        f"Every ratio value that occurs inside the Indeterminate band\n"
        f"all {n_total:,} Indeterminate visits that have a computable ratio",
        fontsize=11.5, fontweight="bold", loc="left", pad=10,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    # Boundary markers, drawn as annotated edges of the band.
    for thr, name, colour, off in (
        (pigd_threshold, f"PIGD boundary {pigd_threshold:.2f}", STATE_COLORS["PIGD"], -0.62),
        (td_threshold, f"TD boundary {td_threshold:.2f}", STATE_COLORS["TD"], len(ratios) - 0.42),
    ):
        ax.axhline(off, color=colour, linewidth=1.6, linestyle="--")
        ax.text(
            max(total) * 1.26, off, name, ha="right", va="bottom",
            fontsize=9, color=colour, fontweight="bold",
        )

    # Parked in the empty middle-right; the two tall bars own the top and bottom.
    leg = ax.legend(frameon=False, fontsize=9, loc="center right",
                    bbox_to_anchor=(1.0, 0.55))
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- right: where each group is at the following visit
    groups = list(next_state.index)
    yy = np.arange(len(groups))
    base = np.zeros(len(groups))
    for state in STATE_ORDER:
        w = next_state[state].to_numpy(dtype=float)
        ax2.barh(
            yy, w, left=base, height=0.55,
            color=STATE_COLORS[state], edgecolor=SURFACE, linewidth=1.4,
        )
        for k, (v, b) in enumerate(zip(w, base)):
            if v > 0.10:
                ax2.text(
                    b + v / 2, k, f"{v:.0%}", ha="center", va="center",
                    color="#ffffff", fontsize=10, fontweight="bold",
                )
        base += w

    for k, state in enumerate(STATE_ORDER):
        ax2.text(
            0.01 + k * 0.34, len(groups) - 0.35, state,
            fontsize=9.5, color=STATE_COLORS[state], fontweight="bold",
        )

    ax2.set_yticks(yy)
    ax2.set_yticklabels(groups, fontsize=9.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.7, len(groups) - 0.1)
    ax2.set_xlabel("share of patients at the next visit", color=INK_MUTED, fontsize=10)
    n_next = int(sum(int(g.split("n = ")[-1]) for g in groups))
    ax2.set_title(
        f"Where they are at the following visit\n"
        f"the {n_next:,} of those visits that are followed by another visit",
        fontsize=11.5, fontweight="bold", loc="left", pad=10,
    )
    ax2.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.text(
        0.008, -0.055,
        "The ratio is an average of 11 tremor items divided by an average of 5 gait items, so only a few values can "
        "land inside a band 0.25 wide.\n"
        f"Left: {n_total:,} of the 441 Indeterminate visits; the other 60 have both subscores at zero, so no ratio "
        f"exists. 78% sit on one of the two edges.\n"
        f"Right: the {n_next:,} left-hand visits that have a following labelled visit; the remaining "
        f"{n_total - n_next} were a patient's last.",
        fontsize=8.5, color=INK_MUTED,
    )
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  2. Patient-level course, every patient
# ---------------------------------------------------------------------------
def patient_state_carpet(
    panel: pd.DataFrame,
    path: Path,
    names: dict,
    max_years: float = 14.0,
    bin_years: float = 0.5,
) -> tuple[Path, int]:
    """One row per patient, coloured by the subtype recorded at that time.

    A patient's row is filled from each visit up to their next visit, so the
    colour shows the state they were last recorded in. Rows are ordered by
    starting subtype and then by how much of the follow-up was spent in PIGD,
    which brings the structure out without hiding any patient.

    Patients with fewer than two labelled visits have no course to show and are
    skipped. Returns the path and the number of rows actually drawn, so the
    caller can check it against the population it believes it passed in.
    """
    df = panel[panel["subtype"].notna()].sort_values(["PATNO", "years_from_baseline"])
    edges = np.arange(0.0, max_years + bin_years, bin_years)
    n_bins = len(edges) - 1

    code = {name: k for k, name in enumerate(STATE_ORDER)}
    rows, meta = [], []

    for patno, g in df.groupby("PATNO", sort=False):
        t = g["years_from_baseline"].to_numpy(dtype=float)
        s = g["subtype"].map(names).to_numpy()
        if len(t) < 2:
            continue
        row = np.full(n_bins, np.nan)
        for k in range(len(t)):
            stop = t[k + 1] if k + 1 < len(t) else t[k] + bin_years
            lo = int(np.searchsorted(edges, t[k], side="right") - 1)
            hi = int(np.searchsorted(edges, stop, side="right") - 1)
            lo, hi = max(lo, 0), min(max(hi, lo), n_bins - 1)
            row[lo : hi + 1] = code[s[k]]
        rows.append(row)
        observed = row[~np.isnan(row)]
        meta.append(
            {
                "PATNO": patno,
                "baseline": code[s[0]],
                "share_pigd": float((observed == code["PIGD"]).mean()) if len(observed) else 0.0,
                "span": float(np.count_nonzero(~np.isnan(row))),
            }
        )

    mat = np.vstack(rows)
    order = (
        pd.DataFrame(meta)
        .sort_values(["baseline", "share_pigd", "span"], ascending=[True, True, False])
        .index.to_numpy()
    )
    mat = mat[order]

    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    ax.set_facecolor(SURFACE)
    cmap = ListedColormap([STATE_COLORS[s] for s in STATE_ORDER])
    cmap.set_bad(SURFACE)

    ax.imshow(
        np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5,
        interpolation="nearest",
        extent=(0.0, max_years, len(mat), 0),
    )

    # Boundaries between the three starting-subtype blocks.
    base_sorted = pd.DataFrame(meta).loc[order, "baseline"].to_numpy()
    for k, state in enumerate(STATE_ORDER):
        idx = np.where(base_sorted == k)[0]
        if not len(idx):
            continue
        if idx[0] > 0:
            ax.axhline(idx[0], color=INK, linewidth=1.2)
        ax.text(
            max_years * 0.995, (idx[0] + idx[-1]) / 2,
            f"started {state}\nn = {len(idx):,}",
            ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.82,
                      boxstyle="round,pad=0.3"),
        )

    ax.set_xlim(0, max_years)
    ax.set_xlabel("years since first visit", color=INK_MUTED, fontsize=10)
    ax.set_ylabel(f"individual patients (n = {len(mat):,}), one row each",
                  color=INK_MUTED, fontsize=10)
    ax.set_yticks([])
    ax.set_title(
        "What happens to each patient: every patient with two or more labelled visits",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)

    ax.legend(
        handles=[Patch(facecolor=STATE_COLORS[s], label=s) for s in STATE_ORDER],
        frameon=False, fontsize=9.5, loc="upper center",
        bbox_to_anchor=(0.5, -0.075), ncol=3, labelcolor=INK,
    )

    fig.text(
        0.008, -0.115,
        f"Each row is one patient with at least two labelled visits (n = {len(mat):,}). Colour is the subtype "
        "recorded at their most recent visit; blank means\nfollow-up had ended. Rows are grouped by starting "
        "subtype, then ordered by how much of follow-up was recorded as PIGD. This figure shows every\npatient "
        "in the analysis: nothing is averaged and no patient is hidden.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path), len(mat)


# ---------------------------------------------------------------------------
#  3. A readable sample: one line per patient, with the actual visits marked
# ---------------------------------------------------------------------------
def patient_timelines(
    panel: pd.DataFrame,
    patients: list,
    path: Path,
    names: dict,
    annotations: dict | None = None,
    coverage: str = "",
) -> Path:
    """Individual timelines for a chosen set of patients.

    The carpet plot shows everyone but no single course in detail; this shows a
    handful of courses in full, with every visit marked.
    """
    df = panel[panel["subtype"].notna()].sort_values(["PATNO", "years_from_baseline"])
    annotations = annotations or {}

    fig, ax = plt.subplots(figsize=(10.0, 0.46 * len(patients) + 2.0))
    _style(ax)

    for row, patno in enumerate(patients):
        g = df[df["PATNO"] == patno]
        t = g["years_from_baseline"].to_numpy(dtype=float)
        s = g["subtype"].map(names).to_numpy()
        y = len(patients) - row - 1

        ax.plot(t, np.full(len(t), y), color=FAINT, linewidth=1.4, zorder=1)
        for k in range(len(t) - 1):
            ax.plot(
                [t[k], t[k + 1]], [y, y],
                color=STATE_COLORS[s[k]], linewidth=4.2, solid_capstyle="butt", zorder=2,
            )
        # A short stub for the final visit, so the state the course *ends* in is
        # visible as a bar rather than only as a dot.
        if len(t) > 1:
            stub = 0.22 * float(np.median(np.diff(t)))
            ax.plot(
                [t[-1], t[-1] + stub], [y, y],
                color=STATE_COLORS[s[-1]], linewidth=4.2, solid_capstyle="butt", zorder=2,
            )
        ax.scatter(t, np.full(len(t), y), s=34, zorder=3,
                   c=[STATE_COLORS[x] for x in s], edgecolor=SURFACE, linewidth=1.2)

        note = annotations.get(patno, "")
        ax.text(-0.35, y, note, ha="right", va="center", fontsize=9, color=INK_MUTED)

    ax.set_yticks([])
    ax.set_ylim(-0.8, len(patients) - 0.2)
    ax.set_xlim(-0.35, df["years_from_baseline"].max() * 0.02 + max(
        df[df["PATNO"].isin(patients)]["years_from_baseline"].max(), 1.0))
    ax.set_xlabel("years since first visit", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "Individual patient courses, one line per patient",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("left",):
        ax.spines[side].set_visible(False)

    ax.legend(
        handles=[Patch(facecolor=STATE_COLORS[s], label=s) for s in STATE_ORDER],
        frameon=False, fontsize=9.5, loc="upper center",
        bbox_to_anchor=(0.5, -0.11), ncol=3, labelcolor=INK,
    )
    fig.text(
        0.008, -0.205,
        "Dots are actual visits; the coloured bar between two dots is the subtype recorded at the earlier one.\n"
        "One patient per course, chosen as the patient with the typical number of visits for that course rather "
        "than the longest followed,\nso the lengths shown here are representative. The count on the left is how "
        f"many patients follow that course, not how many are drawn.\n{coverage}",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  4. Conversion in both directions, side by side
# ---------------------------------------------------------------------------
def conversion_both_directions(
    to_pigd: dict[str, pd.DataFrame],
    to_td: dict[str, pd.DataFrame],
    path: Path,
    min_at_risk: int = 25,
    label_at: float = 5.0,
) -> Path:
    """Cumulative incidence of conversion to PIGD and to TD on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), sharey=True)
    styles = {
        "any single visit": dict(color="#eb6834", linestyle="--"),
        "sustained (2 consecutive visits)": dict(color="#2a78d6", linestyle="-"),
    }

    panels = [
        (axes[0], to_pigd, "PIGD", "patients not PIGD at their first visit"),
        (axes[1], to_td, "TD", "patients not TD at their first visit"),
    ]
    cutoff = min(
        float(km.loc[km["n_at_risk"] >= min_at_risk, "time"].max())
        for curves in (to_pigd, to_td) for km in curves.values()
    )

    for ax, curves, target, risk_note in panels:
        _style(ax)
        for label, km in curves.items():
            vis = km[km["time"] <= cutoff]
            t = vis["time"].to_numpy()
            inc = 1 - vis["survival"].to_numpy()
            se = vis["se"].to_numpy()
            st = styles.get(label, dict(color=INK))
            ax.step(t, inc, where="post", linewidth=2, label=label, **st)
            ax.fill_between(
                t, np.clip(inc - 1.96 * se, 0, 1), np.clip(inc + 1.96 * se, 0, 1),
                step="post", alpha=0.13, color=st["color"], linewidth=0,
            )
            at_h = inc[t <= label_at]
            if len(at_h):
                ax.plot([label_at], [at_h[-1]], "o", color=st["color"], markersize=8, zorder=5)
                ax.annotate(
                    f"{at_h[-1]:.0%}", xy=(label_at, at_h[-1]), xytext=(8, -1),
                    textcoords="offset points", color=st["color"],
                    fontsize=10.5, fontweight="bold", va="center",
                )

        n0 = int(list(curves.values())[0]["n_at_risk"].iloc[0])
        ax.axvline(label_at, color=GRID, linewidth=1.0, zorder=0)
        ax.set_xlim(0, cutoff)
        ax.set_ylim(0, 1)
        ax.set_xlabel("years since first visit", color=INK_MUTED, fontsize=10)
        ax.set_title(
            f"Conversion to {target}\n{risk_note}, n = {n0:,}",
            fontsize=11.5, fontweight="bold", loc="left", pad=10,
        )
        ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("cumulative probability of conversion", color=INK_MUTED, fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(
        0.008, -0.075,
        f"Same method on both sides: Kaplan-Meier, censored at the last labelled visit, truncated at year "
        f"{cutoff:.0f} where fewer than {min_at_risk} patients remain.\nThe dot marks year {label_at:.0f}. "
        "'Sustained' requires the new subtype to still be recorded at the following visit.",
        fontsize=8.5, color=INK_MUTED,
    )
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  5. The raw ratio, before any model
# ---------------------------------------------------------------------------
def raw_ratio_progression(
    panel: pd.DataFrame,
    path: Path,
    td_threshold: float = 1.15,
    pigd_threshold: float = 0.90,
    y_max: float = 8.0,
    max_years: float = 14.0,
) -> Path:
    """Every observed ratio against time, with per-patient lines drawn through.

    No model, no smoothing of individual patients. The only summary drawn is a
    yearly median of the observations actually present in that year.
    """
    df = panel[panel["ratio"].notna()].sort_values(["PATNO", "years_from_baseline"])
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(10.4, 6.4), height_ratios=[5, 1], sharex=True
    )
    _style(ax)
    _style(ax_n)

    # Indeterminate band and the two thresholds.
    ax.axhspan(pigd_threshold, td_threshold, color=STATE_COLORS["Indeterminate"],
               alpha=0.16, zorder=0)
    ax.axhline(td_threshold, color=STATE_COLORS["TD"], linewidth=1.5, linestyle="--", zorder=1)
    ax.axhline(pigd_threshold, color=STATE_COLORS["PIGD"], linewidth=1.5, linestyle="--", zorder=1)

    for _, g in df.groupby("PATNO", sort=False):
        if len(g) < 2:
            continue
        ax.plot(
            g["years_from_baseline"].to_numpy(), g["ratio"].to_numpy(),
            color=FAINT, linewidth=0.5, alpha=0.55, zorder=2,
        )

    clipped = df["ratio"].to_numpy() > y_max
    ax.scatter(
        df["years_from_baseline"].to_numpy(), np.minimum(df["ratio"].to_numpy(), y_max),
        s=5, color="#2a78d6", alpha=0.30, linewidth=0, zorder=3,
    )

    # Raw yearly median: a summary of the observations, not a fitted line.
    yr = df["years_from_baseline"].round().astype(int)
    med = df.groupby(yr)["ratio"].agg(["median", "count"])
    med = med[(med["count"] >= 25) & (med.index <= max_years)]
    ax.plot(med.index.to_numpy(), med["median"].to_numpy(), color=INK,
            linewidth=2.4, zorder=5, label="median of the observations in each year")
    ax.scatter(med.index.to_numpy(), med["median"].to_numpy(), s=26, color=INK, zorder=6)

    ax.text(max_years * 0.995, td_threshold + 0.16, f"TD boundary {td_threshold:.2f}",
            ha="right", fontsize=9, color=STATE_COLORS["TD"], fontweight="bold")
    ax.text(max_years * 0.995, pigd_threshold - 0.34, f"PIGD boundary {pigd_threshold:.2f}",
            ha="right", fontsize=9, color=STATE_COLORS["PIGD"], fontweight="bold")
    ax.text(0.12, (td_threshold + pigd_threshold) / 2, "Indeterminate", fontsize=8.5,
            va="center", color="#a8461c", fontweight="bold")

    ax.set_xlim(-0.2, max_years)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("TD/PIGD ratio recorded at the visit", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "The raw ratio at every visit, with each patient's own visits joined",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.grid(color=GRID, linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)

    counts = df.groupby(yr).size()
    counts = counts[counts.index <= max_years]
    ax_n.bar(counts.index.to_numpy(), counts.to_numpy(), color=FAINT,
             edgecolor=SURFACE, linewidth=1.6)
    ax_n.set_ylabel("visits", color=INK_MUTED, fontsize=9)
    ax_n.set_xlabel("years since first visit", color=INK_MUTED, fontsize=10)
    ax_n.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax_n.set_axisbelow(True)

    fig.text(
        0.008, -0.055,
        f"{len(df):,} visits from {df['PATNO'].nunique():,} patients. Grey lines join visits belonging to the same "
        f"patient. {int(clipped.sum()):,} points above {y_max:.0f} are drawn at the top edge.\n"
        "The black line is the median of the observations present in each year, not a fitted trend, and is "
        "affected by which patients are still attending.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


def ratio_within_patient(panel: pd.DataFrame, path: Path,
                         td_threshold: float = 1.15,
                         pigd_threshold: float = 0.90) -> Path:
    """How far a patient's own ratio moves between visits, against the band width."""
    df = panel[panel["ratio"].notna()].sort_values(["PATNO", "years_from_baseline"])
    steps = (
        df.groupby("PATNO")["ratio"]
        .apply(lambda s: s.diff().abs().dropna())
        .reset_index(drop=True)
    )
    band = td_threshold - pigd_threshold

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    _style(ax)
    bins = np.arange(0, 6.01, 0.125)
    ax.hist(np.minimum(steps.to_numpy(), 6.0), bins=bins, color="#2a78d6",
            edgecolor=SURFACE, linewidth=0.6)

    ax.axvline(band, color=STATE_COLORS["Indeterminate"], linewidth=2.2)
    top = ax.get_ylim()[1]
    ax.annotate(
        f"width of the whole\nIndeterminate band ({band:.2f})",
        xy=(band, top * 0.55), xytext=(1.55, top * 0.90),
        fontsize=9.5, color="#a8461c", fontweight="bold", va="top",
        arrowprops=dict(arrowstyle="-", color=STATE_COLORS["Indeterminate"], linewidth=1.2),
    )

    share = float((steps > band).mean())
    ax.text(
        0.985, 0.62,
        f"{share:.0%} of consecutive visits\nmove the ratio by more than\nthe entire band width",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        color=INK, fontweight="bold",
    )

    ax.set_xlim(0, 6)
    ax.set_xlabel("change in the ratio between one visit and the next",
                  color=INK_MUTED, fontsize=10)
    ax.set_ylabel("number of consecutive visit pairs", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "How much one patient's own ratio moves between visits",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.008, -0.08,
        f"{len(steps):,} consecutive pairs of visits from the same patient with a ratio at both ends. Changes above "
        "6 are drawn in the last bin.\nThis is the same patient measured twice, so it is variation within a person "
        "rather than variation between people.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  6. The three starting subtypes on one set of axes
# ---------------------------------------------------------------------------
def combined_progression(
    fit,
    path: Path,
    max_years: float = 10.0,
    label_at: float = 5.0,
) -> Path:
    """Predicted subtype over time for all three starting subtypes, combined.

    Colour is the subtype being predicted; line style is the subtype the patient
    starts in. This is the same quantity as the three-panel version, drawn once
    so the three starting points can be compared directly.
    """
    times = np.linspace(0.0, max_years, 241)
    dashes = {"TD": (None, None), "Indeterminate": (5, 2.2), "PIGD": (1.4, 2.0)}

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    _style(ax)

    at = int(np.argmin(np.abs(times - label_at)))
    labels = []
    for start in STATE_ORDER:
        probs = np.array([fit.P(t).loc[start].to_numpy() for t in times])
        for k, target in enumerate(STATE_ORDER):
            ax.plot(
                times, probs[:, k],
                color=STATE_COLORS[target], linewidth=2.0,
                dashes=dashes[start], solid_capstyle="round",
                alpha=0.95 if start == target else 0.8,
            )
            labels.append((float(probs[at, k]), STATE_COLORS[target]))

    # Nine curves cross near year 5, so the year-5 readings collide. Push them
    # apart vertically and draw a leader back to the true value.
    labels.sort(key=lambda r: r[0])
    gap, placed = 0.058, []
    for value, colour in labels:
        y = value if not placed else max(value, placed[-1][0] + gap)
        placed.append((y, value, colour))
    for y, value, colour in placed:
        ax.plot([label_at, label_at + 0.34], [value, y], color=colour,
                linewidth=0.8, alpha=0.7, zorder=4)
        ax.text(label_at + 0.40, y, f"{value:.0%}", fontsize=8.5,
                color=colour, va="center", ha="left", fontweight="bold", zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.14"))

    ax.axvline(label_at, color=GRID, linewidth=1.2, zorder=0)
    ax.text(label_at + 0.1, 1.01, f"year {label_at:.0f}", fontsize=9, color=INK_MUTED)

    ax.set_xlim(0, max_years)
    ax.set_ylim(0, 1)
    ax.set_xlabel("years from now", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("probability of being in that subtype", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "Predicted subtype over time, all three starting subtypes on one axis",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.grid(color=GRID, linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    colour_keys = [Patch(facecolor=STATE_COLORS[s], label=f"ends {s}") for s in STATE_ORDER]
    style_keys = [
        Line2D([0], [0], color=INK_MUTED, linewidth=2.0, dashes=dashes[s], label=f"starts {s}")
        for s in STATE_ORDER
    ]
    leg = ax.legend(handles=colour_keys + style_keys, frameon=False, fontsize=9,
                    loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(
        0.008, -0.155,
        "Colour is the subtype being predicted; line style is the subtype the patient is in now. Percentages are "
        f"read at year {label_at:.0f}.\nSame numbers as the three-panel version, drawn together so the three "
        "starting points can be compared without switching figures.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)
