"""Figures for the subtype transition analysis.

Palette
-------
State identity uses categorical slots 1-3 in fixed order (TD, Indeterminate,
PIGD). Validated for all-pairs CVD separation on the light surface: worst pair
ΔE 9.2 deutan, 24.0 normal vision. The aqua slot sits at 2.74:1 against the
surface, below the 3:1 bar, so every figure carries direct labels and every
figure writes a companion CSV — the relief rule for a contrast warning.

Transition *probability* is magnitude, not identity, so the matrix heatmap uses
a single-hue sequential blue ramp rather than the categorical hues.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

STATE_COLORS = {"TD": "#2a78d6", "Indeterminate": "#eb6834", "PIGD": "#1baf7a"}
STATE_ORDER = ["TD", "Indeterminate", "PIGD"]

# Sequential blue, steps 100 -> 700 from the reference ramp.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def _style(ax) -> None:
    """Recessive axes and grid; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=1.0)
    ax.title.set_color(INK)


def _finish(fig, path: Path) -> Path:
    fig.patch.set_facecolor(SURFACE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
#  1. Transition matrix
# ---------------------------------------------------------------------------
def transition_heatmap(
    probs: pd.DataFrame, counts: pd.DataFrame, path: Path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    _style(ax)

    data = probs.reindex(index=STATE_ORDER, columns=STATE_ORDER).to_numpy(dtype=float)
    n = counts.reindex(index=STATE_ORDER, columns=STATE_ORDER).to_numpy(dtype=float)

    ax.imshow(data, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto")

    for i in range(len(STATE_ORDER)):
        for j in range(len(STATE_ORDER)):
            if np.isnan(data[i, j]):
                continue
            # Ink flips to white only where the fill is dark enough to need it.
            colour = "#ffffff" if data[i, j] > 0.55 else INK
            ax.text(
                j, i - 0.10, f"{data[i, j]:.3f}",
                ha="center", va="center", color=colour, fontsize=13, fontweight="bold",
            )
            ax.text(
                j, i + 0.20, f"n = {int(n[i, j]):,}",
                ha="center", va="center", color=colour, fontsize=9, alpha=0.85,
            )
            # 2px surface gap between cells.
            ax.add_patch(
                Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SURFACE, linewidth=2)
            )

    ax.set_xticks(range(len(STATE_ORDER)))
    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_xticklabels(STATE_ORDER, fontsize=10)
    ax.set_yticklabels(STATE_ORDER, fontsize=10)
    ax.set_xlabel("state at next visit", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("state at current visit", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, loc="left")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  2. Prevalence over follow-up
# ---------------------------------------------------------------------------
def prevalence_area(prevalence: pd.DataFrame, path: Path, max_year: int = 12) -> Path:
    df = prevalence[prevalence.index <= max_year]
    years = df.index.to_numpy()
    cols = [f"prop_{s}" for s in STATE_ORDER]
    values = [df[c].to_numpy() for c in cols]

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(8.4, 5.4), height_ratios=[4, 1], sharex=True
    )
    _style(ax)
    _style(ax_n)

    ax.stackplot(
        years, *values,
        colors=[STATE_COLORS[s] for s in STATE_ORDER],
        edgecolor=SURFACE, linewidth=2,
    )

    # Direct labels inside each band at its widest point, held clear of the
    # axes so a band that peaks at year 0 is not clipped by the y-axis.
    cumulative = np.cumsum(values, axis=0)
    span = years.max() - years.min()
    for k, state in enumerate(STATE_ORDER):
        band = values[k]
        idx = int(np.argmax(band))
        lower = cumulative[k - 1][idx] if k > 0 else 0.0
        x = float(np.clip(years[idx], years.min() + 0.10 * span, years.max() - 0.10 * span))
        ax.text(
            x, lower + band[idx] / 2, state,
            ha="center", va="center", color="#ffffff", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(years.min(), years.max())
    ax.set_ylabel("share of assessed patients", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "Motor subtype prevalence across follow-up",
        fontsize=12, fontweight="bold", pad=12, loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    ax_n.bar(years, df["n_total"].to_numpy(), color="#c9c8c4", edgecolor=SURFACE, linewidth=2)
    ax_n.set_ylabel("n assessed", color=INK_MUTED, fontsize=9)
    ax_n.set_xlabel("years since baseline", color=INK_MUTED, fontsize=10)
    ax_n.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax_n.set_axisbelow(True)

    fig.text(
        0.008, -0.04,
        "Attrition is heavy and unlikely to be random: the n panel is part of the reading, "
        "not a footnote.\nA prevalence shift mixes genuine progression with selective dropout.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  3. Transition probability by elapsed interval
# ---------------------------------------------------------------------------
def transitions_by_interval(
    table: pd.DataFrame,
    path: Path,
    highlight: list[tuple[str, str]],
    small_n: int = 30,
) -> Path:
    """Selected transitions against elapsed interval.

    Each *series* is a from→to pair, so colour is assigned per series from the
    categorical slots in fixed order. Colouring by destination state instead
    would paint TD→PIGD and Indeterminate→PIGD identically.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    _style(ax)

    series_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    bins = list(dict.fromkeys(table["interval"]))
    x = np.arange(len(bins))
    width = 0.8 / max(len(highlight), 1)

    bin_totals = {
        b: int(table[table["interval"] == b].groupby("from_state")["n_from"].first().sum())
        for b in bins
    }

    ymax, flagged = 0.0, False
    for k, (frm, to) in enumerate(highlight):
        sub = table[(table["from_state"] == frm) & (table["to_state"] == to)]
        sub = sub.set_index("interval").reindex(bins)
        vals = sub["probability"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        ns = sub["n_from"].to_numpy(dtype=float)
        pos = x - 0.4 + width * (k + 0.5)
        sparse = ns < small_n

        ax.bar(
            pos, vals, width * 0.86,
            color=series_colors[k % len(series_colors)],
            edgecolor=SURFACE, linewidth=2,
            alpha=1.0, label=f"{frm} → {to}",
        )
        # Sparse cells are washed out so the eye discounts them automatically.
        ax.bar(
            np.where(sparse, pos, np.nan), np.where(sparse, vals, np.nan), width * 0.86,
            color=SURFACE, edgecolor=SURFACE, linewidth=2, alpha=0.45,
        )
        ax.errorbar(
            pos, vals, yerr=[vals - lo, hi - vals],
            fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3, alpha=0.75,
        )
        for xi, v, h, is_sparse in zip(pos, vals, hi, sparse):
            if np.isnan(v):
                continue
            mark = "†" if is_sparse else ""
            flagged = flagged or bool(is_sparse)
            ax.text(xi, h + 0.018, f"{v:.2f}{mark}", ha="center", fontsize=8.5, color=INK)
            ymax = max(ymax, h)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\nn = {bin_totals[b]:,}" for b in bins], fontsize=9.5, color=INK_MUTED
    )
    ax.set_xlabel("elapsed interval between visits (years)", color=INK_MUTED, fontsize=10, labelpad=6)
    ax.set_ylabel("transition probability", color=INK_MUTED, fontsize=10)
    ax.set_ylim(0, min(1.0, ymax + 0.12))
    ax.set_title(
        "Transition probability grows with the gap between visits",
        fontsize=12, fontweight="bold", pad=12, loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncols=len(highlight))
    for text in leg.get_texts():
        text.set_color(INK)

    note = (
        "Whiskers are Wilson 95% intervals; n under each bin is the number of transitions observed in it.\n"
        "Pooling these bins into one matrix would conflate progression rate with visit attendance."
    )
    if flagged:
        note += f"\n†  fewer than {small_n} transitions from that state — read the interval, not the point estimate."
    fig.text(0.008, -0.12, note, fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  4. Time to PIGD conversion
# ---------------------------------------------------------------------------
def conversion_curves(
    curves: dict[str, pd.DataFrame],
    path: Path,
    min_at_risk: int = 25,
    label_at: float = 5.0,
    target_name: str = "PIGD",
) -> Path:
    """Cumulative incidence of conversion to `target_name` under two definitions.

    The curves are truncated where the risk set falls below `min_at_risk`. Past
    that point a Kaplan-Meier estimate is driven by a handful of patients and
    every step is a large jump; carrying it to year 14 would put a confident
    number on top of two people. Direct labels are placed at a horizon the data
    actually supports, and the risk set is printed beneath the axis.
    """
    fig, (ax, ax_risk) = plt.subplots(
        2, 1, figsize=(8.0, 5.4), height_ratios=[5, 1], sharex=True
    )
    _style(ax)
    ax_risk.set_facecolor(SURFACE)
    ax_risk.axis("off")

    styles = {
        "any single visit": dict(color="#eb6834", linestyle="--"),
        "sustained (2 consecutive visits)": dict(color="#2a78d6", linestyle="-"),
    }

    cutoff = min(
        float(km.loc[km["n_at_risk"] >= min_at_risk, "time"].max()) for km in curves.values()
    )

    for row, (label, km) in enumerate(curves.items()):
        vis = km[km["time"] <= cutoff]
        t = vis["time"].to_numpy()
        incidence = 1 - vis["survival"].to_numpy()
        se = vis["se"].to_numpy()
        style = styles.get(label, dict(color=INK))

        ax.step(t, incidence, where="post", linewidth=2, label=label, **style)
        ax.fill_between(
            t,
            np.clip(incidence - 1.96 * se, 0, 1),
            np.clip(incidence + 1.96 * se, 0, 1),
            step="post", alpha=0.13, color=style["color"], linewidth=0,
        )

        at_h = incidence[t <= label_at]
        if len(at_h):
            ax.annotate(
                f"{at_h[-1]:.0%} by year {label_at:.0f}",
                xy=(label_at, at_h[-1]), xytext=(8, -2), textcoords="offset points",
                color=style["color"], fontsize=10, fontweight="bold", va="center",
            )
            ax.plot([label_at], [at_h[-1]], "o", color=style["color"], markersize=8, zorder=5)

        # Risk set beneath the axis, one row per curve.
        ticks = np.arange(0, cutoff + 0.01, 2.0)
        for tick in ticks:
            n = int(km.loc[km["time"] <= tick, "n_at_risk"].iloc[-1]) if (km["time"] <= tick).any() else 0
            ax_risk.text(
                tick, -row * 0.9, f"{n:,}",
                ha="center", va="center", fontsize=8.5, color=style["color"],
            )
        ax_risk.text(
            -0.55, -row * 0.9, label.split(" (")[0], ha="right", va="center",
            fontsize=8.5, color=style["color"],
        )

    ax.axvline(label_at, color=GRID, linewidth=1.0, zorder=0)
    ax.set_xlim(0, cutoff)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"cumulative probability of {target_name} conversion", color=INK_MUTED, fontsize=10)
    ax.set_title(
        f"Time to first {target_name} conversion, by how strictly conversion is defined",
        fontsize=12, fontweight="bold", pad=12, loc="left",
    )
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK)

    ax_risk.set_xlim(0, cutoff)
    ax_risk.set_ylim(-1.5, 0.7)
    ax_risk.text(-0.55, 0.45, "at risk", ha="right", va="center", fontsize=8.5,
                 color=INK, fontweight="bold")
    ax_risk.set_xlabel("years since baseline", color=INK_MUTED, fontsize=10)
    ax_risk.axis("on")
    for side in ("top", "right", "left"):
        ax_risk.spines[side].set_visible(False)
    ax_risk.spines["bottom"].set_color(GRID)
    ax_risk.set_yticks([])
    ax_risk.tick_params(colors=INK_MUTED, labelsize=9, length=3)

    fig.text(
        0.008, -0.04,
        f"Kaplan-Meier, censored at last labelled visit; risk set is patients not already {target_name} at baseline.\n"
        f"Truncated at year {cutoff:.0f}, where the risk set falls below {min_at_risk}. The gap between the curves is the\n"
        "share of apparent conversions produced by a single assessment that did not hold at the next visit.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  5. Reversion after an observed change
# ---------------------------------------------------------------------------
def reversion_bars(summary: pd.DataFrame, path: Path) -> Path:
    df = summary.sort_values("p_revert", ascending=True)
    labels = [f"{r.from_state} → {r.to_state}" for r in df.itertuples()]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(df) + 2.2))
    _style(ax)

    ax.barh(
        y, df["p_revert"].to_numpy(),
        color=[STATE_COLORS[s] for s in df["from_state"]],
        edgecolor=SURFACE, linewidth=2, height=0.68,
    )
    for yi, (p, n, k) in enumerate(zip(df["p_revert"], df["n"], df["n_reverted"])):
        ax.text(p + 0.012, yi, f"{p:.0%}   ({int(k)}/{int(n)})", va="center", fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, min(1.0, float(df["p_revert"].max()) + 0.22))
    ax.set_xlabel("share returning to the origin state at the next visit", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "How often an observed subtype change reverses immediately",
        fontsize=12, fontweight="bold", pad=12, loc="left",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    fig.text(
        0.008, -0.10,
        "Bar colour marks the origin state. A high rate means the observed label is a noisy read of an\n"
        "underlying state rather than a durable change — the premise the latent-variable model rests on.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  6. Pathway flow
# ---------------------------------------------------------------------------
def pathway_flow(
    frames: list[pd.DataFrame], column_labels: list[str], path: Path, n_patients: int
) -> Path:
    """Alluvial flow across fixed follow-up points.

    `frames[k]` is a crosstab of state at column k against state at column k+1.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_facecolor(SURFACE)
    ax.axis("off")

    n_cols = len(column_labels)
    gap, box_w = 0.045, 0.075
    x_positions = np.linspace(0.0, 1.0, n_cols)

    # Column totals from the first frame's rows, then each frame's columns.
    totals = [frames[0].sum(axis=1).reindex(STATE_ORDER).fillna(0)]
    for f in frames:
        totals.append(f.sum(axis=0).reindex(STATE_ORDER).fillna(0))

    # Vertical extents for each state box in each column.
    extents = []
    for col_total in totals:
        total = col_total.sum()
        spans, cursor = {}, 0.0
        usable = 1.0 - gap * (len(STATE_ORDER) - 1)
        for s in STATE_ORDER:
            h = (col_total[s] / total) * usable if total else 0.0
            spans[s] = (cursor, cursor + h)
            cursor += h + gap
        extents.append(spans)

    # Ribbons, drawn before boxes so the boxes sit on top.
    for k, frame in enumerate(frames):
        x0, x1 = x_positions[k] + box_w / 2, x_positions[k + 1] - box_w / 2
        out_cursor = {s: extents[k][s][0] for s in STATE_ORDER}
        in_cursor = {s: extents[k + 1][s][0] for s in STATE_ORDER}
        col_out = frame.sum(axis=1)
        col_in = frame.sum(axis=0)

        for src in STATE_ORDER:
            if src not in frame.index:
                continue
            src_h = extents[k][src][1] - extents[k][src][0]
            for dst in STATE_ORDER:
                if dst not in frame.columns:
                    continue
                v = float(frame.loc[src, dst])
                if v <= 0:
                    continue
                dst_h = extents[k + 1][dst][1] - extents[k + 1][dst][0]
                h0 = src_h * v / col_out[src] if col_out[src] else 0
                h1 = dst_h * v / col_in[dst] if col_in[dst] else 0
                y0a, y0b = out_cursor[src], out_cursor[src] + h0
                y1a, y1b = in_cursor[dst], in_cursor[dst] + h1
                out_cursor[src] += h0
                in_cursor[dst] += h1

                cx = (x0 + x1) / 2
                verts = [
                    (x0, y0a), (cx, y0a), (cx, y1a), (x1, y1a),
                    (x1, y1b), (cx, y1b), (cx, y0b), (x0, y0b), (x0, y0a),
                ]
                codes = [
                    MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                    MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                    MplPath.CLOSEPOLY,
                ]
                ax.add_patch(
                    PathPatch(
                        MplPath(verts, codes),
                        facecolor=STATE_COLORS[src], edgecolor="none",
                        alpha=0.30 if src == dst else 0.52,
                    )
                )

    # Boxes carry counts only. State names live in the legend, so a long word
    # like "Indeterminate" cannot overflow a narrow box or run off the canvas.
    for k, spans in enumerate(extents):
        total = totals[k].sum()
        for s in STATE_ORDER:
            y0, y1 = spans[s]
            if y1 - y0 <= 0:
                continue
            ax.add_patch(
                Rectangle(
                    (x_positions[k] - box_w / 2, y0), box_w, y1 - y0,
                    facecolor=STATE_COLORS[s], edgecolor=SURFACE, linewidth=2,
                )
            )
            count, share = totals[k][s], totals[k][s] / total
            if y1 - y0 > 0.08:
                ax.text(
                    x_positions[k], (y0 + y1) / 2, f"{count:.0f}\n{share:.0%}",
                    ha="center", va="center", fontsize=9, color="#ffffff", fontweight="bold",
                )
            else:
                # Too short for interior text: label outside, away from ribbons.
                outside = x_positions[k] - box_w / 2 - 0.012 if k == 0 else x_positions[k] + box_w / 2 + 0.012
                ax.text(
                    outside, (y0 + y1) / 2, f"{count:.0f} ({share:.0%})",
                    ha="right" if k == 0 else "left", va="center",
                    fontsize=8.5, color=STATE_COLORS[s], fontweight="bold",
                )

    for k, label in enumerate(column_labels):
        ax.text(x_positions[k], -0.07, label, ha="center", fontsize=10.5, color=INK, fontweight="bold")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS[s], edgecolor=SURFACE) for s in STATE_ORDER
    ]
    leg = ax.legend(
        handles, STATE_ORDER, frameon=False, fontsize=9.5, ncols=3,
        loc="upper center", bbox_to_anchor=(0.5, 1.10),
    )
    for text in leg.get_texts():
        text.set_color(INK)

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.14, 1.06)
    ax.set_title(
        f"Subtype pathways, patients assessed at all three points (n = {n_patients})",
        fontsize=12, fontweight="bold", pad=26, loc="left", color=INK,
    )
    fig.text(
        0.008, -0.02,
        "Ribbon colour marks the state being left. Restricted to complete cases, so the picture is free of\n"
        "attrition but not of the selection that surviving to Year 4 with a full record implies.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)
