"""Figures for the Weeks 5-7 progression model.

Palette and conventions follow :mod:`ppmi_subtype.plots`: categorical slots 1-3
in fixed order for state identity, validated for all-pairs CVD separation, with
direct labels on every figure because the aqua slot sits below 3:1 contrast on
the light surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .plots import GRID, INK, INK_MUTED, STATE_COLORS, STATE_ORDER, SURFACE, _finish, _style

SEQUENTIAL_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange",
    ["#fcfcfb", "#fbdcce", "#f7b99e", "#f2966f", "#eb6834", "#c94f22", "#a13d18", "#772c10"],
)


# ---------------------------------------------------------------------------
def progression_curves(fit, path: Path, max_years: float = 10.0, horizons=(1, 2, 3, 5)) -> Path:
    """The deliverable figure: P(subtype at t) for each starting phenotype.

    Three panels, one per current state. Each panel stacks the probability of
    occupying each subtype as a function of elapsed time, which is exactly the
    output the proposal specifies for a clinician.
    """
    times = np.linspace(0.0, max_years, 241)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharey=True)

    for ax, start in zip(axes, STATE_ORDER):
        _style(ax)
        probs = np.array([fit.P(t).loc[start].to_numpy() for t in times])

        ax.stackplot(
            times, *[probs[:, k] for k in range(3)],
            colors=[STATE_COLORS[s] for s in STATE_ORDER],
            edgecolor=SURFACE, linewidth=1.2,
        )

        cumulative = np.cumsum(probs, axis=1)
        for k, state in enumerate(STATE_ORDER):
            band = probs[:, k]
            idx = int(np.argmax(band))
            lower = cumulative[idx, k - 1] if k > 0 else 0.0
            if band[idx] > 0.09:
                x = float(np.clip(times[idx], max_years * 0.14, max_years * 0.86))
                ax.text(x, lower + band[idx] / 2, state, ha="center", va="center",
                        color="#ffffff", fontsize=9.5, fontweight="bold")

        for h in horizons:
            ax.axvline(h, color=SURFACE, linewidth=1.0, alpha=0.55, zorder=3)

        ax.set_xlim(0, max_years)
        ax.set_ylim(0, 1)
        ax.set_title(f"Currently {start}", fontsize=11, fontweight="bold", loc="left", pad=8)
        ax.set_xlabel("years from now", color=INK_MUTED, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("probability of occupying subtype", color=INK_MUTED, fontsize=10)

    fig.suptitle(
        "Predicted motor subtype over time, by current phenotype",
        fontsize=13, fontweight="bold", x=0.008, ha="left", y=1.03,
    )
    fig.text(
        0.008, -0.07,
        "Hidden Markov model: the latent phenotype evolves as a continuous-time Markov chain, so any horizon "
        "can be read off.\nVertical guides mark the 1, 2, 3 and 5 year horizons. Fitted on training patients "
        "only.",
        fontsize=8.5, color=INK_MUTED,
    )
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
def model_comparison(robustness: pd.DataFrame, path: Path) -> Path:
    """Held-out performance of every model across independent splits."""
    models = ["HMM", "DTMC", "CTMC", "Marginal"]
    labels = {
        "HMM": "HMM\n(latent + noise)",
        "DTMC": "Discrete\nMarkov",
        "CTMC": "Continuous-time\nMarkov",
        "Marginal": "Marginal\n(null)",
    }
    colours = {"HMM": "#2a78d6", "DTMC": "#eb6834", "CTMC": "#1baf7a", "Marginal": "#c9c8c4"}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6), width_ratios=[3, 2])
    _style(ax)
    _style(ax2)

    x = np.arange(len(models))
    for i, row in robustness.iterrows():
        ax.plot(x, [row[m] for m in models], color="#c9c8c4", linewidth=0.9,
                marker="o", markersize=3.5, alpha=0.75, zorder=2)

    for k, m in enumerate(models):
        mean = robustness[m].mean()
        ax.plot([k - 0.24, k + 0.24], [mean, mean], color=colours[m], linewidth=3.5, zorder=4)
        ax.text(k, mean + 0.008, f"{mean:.4f}", ha="center", fontsize=9.5,
                color=INK, fontweight="bold", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models], fontsize=9.5)
    ax.set_ylabel("held-out log-likelihood per transition", color=INK_MUTED, fontsize=10)
    ax.set_title(
        f"Every model, every split (n = {len(robustness)} independent patient-level splits)",
        fontsize=11.5, fontweight="bold", loc="left", pad=10,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    diff = robustness["HMM"] - robustness["DTMC"]
    ax2.barh(np.arange(len(diff)), diff.to_numpy(), color="#2a78d6",
             edgecolor=SURFACE, linewidth=2, height=0.68)
    ax2.axvline(0, color=INK_MUTED, linewidth=1.2)
    ax2.set_yticks(np.arange(len(diff)))
    ax2.set_yticklabels([f"split {i}" for i in robustness["split"]], fontsize=9)
    ax2.set_xlabel("HMM advantage over the discrete chain", color=INK_MUTED, fontsize=10)
    ax2.set_title(
        f"Wins {int((diff > 0).sum())} of {len(diff)}   mean {diff.mean():+.4f}",
        fontsize=11.5, fontweight="bold", loc="left", pad=10,
    )
    ax2.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.text(
        0.008, -0.05,
        "Higher is better. Grey lines join the four models within one split, so a line that never crosses "
        "shows a consistent ranking.",
        fontsize=8.5, color=INK_MUTED,
    )
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
def emission_heatmap(E: pd.DataFrame, path: Path) -> Path:
    """How reliably each true state is recorded by the examination."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    _style(ax)

    data = E.reindex(index=STATE_ORDER, columns=STATE_ORDER).to_numpy(dtype=float)
    ax.imshow(data, cmap=SEQUENTIAL_ORANGE, vmin=0, vmax=1, aspect="auto")

    for i in range(3):
        for j in range(3):
            colour = "#ffffff" if data[i, j] > 0.55 else INK
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color=colour, fontsize=13, fontweight=weight)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=SURFACE, linewidth=2))

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(STATE_ORDER, fontsize=10)
    ax.set_yticklabels(STATE_ORDER, fontsize=10)
    ax.set_xlabel("label actually recorded", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("true latent phenotype", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_title("Estimated misclassification of the clinical label",
                 fontsize=12, fontweight="bold", pad=14, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.text(
        0.008, -0.06,
        "Diagonal entries are the probability an examination records the phenotype correctly.\n"
        "The Indeterminate row sits at the 0.50 floor imposed for identifiability, so it should be read\n"
        "as “at most half of Indeterminate labels are right” rather than as a point estimate.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
def sojourn_comparison(sojourn: pd.DataFrame, path: Path) -> Path:
    """Observed-label persistence against the model's latent persistence."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    _style(ax)

    states = [s for s in STATE_ORDER if s in sojourn.index]
    x = np.arange(len(states))
    width = 0.36

    for k, (col, colour) in enumerate([("CTMC", "#eb6834"), ("HMM (latent)", "#2a78d6")]):
        vals = sojourn.loc[states, col].to_numpy(dtype=float)
        pos = x - width / 2 + k * width
        ax.bar(pos, vals, width * 0.9, color=colour, edgecolor=SURFACE, linewidth=2,
               label="observed label" if col == "CTMC" else "latent phenotype")
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=10)
    ax.set_ylabel("mean years before leaving the state", color=INK_MUTED, fontsize=10)
    ax.set_title("The underlying phenotype is far more stable than its label",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(
        0.008, -0.09,
        "Both are model estimates, not raw counts. Treating the label as the state makes every phenotype look\n"
        "transient; separating measurement error from change recovers a much slower underlying process.",
        fontsize=8.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
def calibration_plot(cal: pd.DataFrame, path: Path, title: str) -> Path:
    """Predicted versus observed rate, with the identity line."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    _style(ax)

    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.6, zorder=1)
    sizes = 40 + 240 * cal["n"] / cal["n"].max()
    ax.scatter(cal["mean_predicted"], cal["observed_rate"], s=sizes,
               color="#2a78d6", edgecolor=SURFACE, linewidth=2, zorder=3)

    for r in cal.itertuples():
        ax.annotate(f"n={int(r.n)}", (r.mean_predicted, r.observed_rate),
                    xytext=(9, -11), textcoords="offset points",
                    fontsize=8.5, color=INK_MUTED)

    top = float(max(cal["mean_predicted"].max(), cal["observed_rate"].max())) + 0.12
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("mean predicted probability", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("observed rate", color=INK_MUTED, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    fig.text(0.008, -0.06,
             "Points on the diagonal are well calibrated; marker area is the number of held-out transitions.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
def growth_trajectories(post, path: Path, max_years: float = 12.0, n_patients: int = 60) -> Path:
    """Population trajectory of the TD/PIGD ratio, with patient-level spread."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), width_ratios=[3, 2])
    _style(ax)
    _style(ax2)

    t = np.linspace(0, max_years, 200)
    rng = np.random.default_rng(0)

    picks = rng.choice(len(post.sigma2), size=n_patients)
    for s in picks:
        b0 = rng.normal(0, np.sqrt(post.tau0_2[s]))
        b1 = rng.normal(0, np.sqrt(post.tau1_2[s]))
        ax.plot(t, (post.beta[s, 0] + b0) + (post.beta[s, 1] + b1) * t,
                color="#c9c8c4", linewidth=0.8, alpha=0.7, zorder=2)

    mean = post.beta[:, 0].mean() + post.beta[:, 1].mean() * t
    draws = np.array([post.beta[s, 0] + post.beta[s, 1] * t for s in range(len(post.sigma2))])
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    ax.fill_between(t, lo, hi, color="#2a78d6", alpha=0.22, linewidth=0, zorder=3)
    ax.plot(t, mean, color="#2a78d6", linewidth=3, zorder=4, label="population trajectory")

    ax.axhline(1.15, color=STATE_COLORS["TD"], linewidth=1.6, linestyle="--", zorder=1)
    ax.axhline(0.90, color=STATE_COLORS["PIGD"], linewidth=1.6, linestyle="--", zorder=1)
    ax.text(max_years * 0.015, 1.22, "TD threshold 1.15", ha="left", fontsize=8.5,
            color=STATE_COLORS["TD"])
    ax.text(max_years * 0.015, 0.66, "PIGD threshold 0.90", ha="left", fontsize=8.5,
            color=STATE_COLORS["PIGD"])
    ax.axhspan(-1.0, 0.0, color="#c9c8c4", alpha=0.35, zorder=0)
    ax.text(max_years * 0.985, -0.55, "ratio < 0 is impossible", ha="right",
            fontsize=8.5, color=INK_MUTED, style="italic")

    ax.set_xlim(0, max_years)
    ax.set_ylim(-1.0, 6.0)
    ax.set_xlabel("years from first visit", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("TD/PIGD ratio", color=INK_MUTED, fontsize=10)
    ax.set_title("Fitted growth model: population trajectory and patient spread",
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK)

    slope = -post.beta[:, 1]
    ax2.hist(slope, bins=40, color="#2a78d6", edgecolor=SURFACE, linewidth=0.8)
    for q, style in ((0.025, ":"), (0.5, "-"), (0.975, ":")):
        ax2.axvline(np.quantile(slope, q), color=INK, linewidth=1.4, linestyle=style)
    ax2.set_xlabel("population decline in ratio, per year", color=INK_MUTED, fontsize=10)
    ax2.set_ylabel("posterior draws", color=INK_MUTED, fontsize=10)
    ax2.set_title(f"Progression rate: {slope.mean():.4f}/yr "
                  f"[{np.quantile(slope,.025):.4f}, {np.quantile(slope,.975):.4f}]",
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.text(0.008, -0.05,
             "Grey lines are 60 patients drawn from the fitted random-effect distribution. The shaded band is the 95% "
             "credible\ninterval for the population mean, which is narrow; the spread between patients is far wider "
             "than the uncertainty in the average.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    return _finish(fig, path)


def hb_vs_hmm(robustness: pd.DataFrame, path: Path) -> Path:
    """Held-out comparison of the two latent-variable models across splits."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    _style(ax)

    x = np.arange(len(robustness))
    width = 0.38
    ax.bar(x - width / 2, robustness["HMM"], width * 0.9, color="#2a78d6",
           edgecolor=SURFACE, linewidth=2, label="HMM (latent states)")
    ax.bar(x + width / 2, robustness["HB growth"], width * 0.9, color="#eb6834",
           edgecolor=SURFACE, linewidth=2, label="Hierarchical Bayesian growth")

    for xi, v in zip(x - width / 2, robustness["HMM"]):
        ax.text(xi, v - 0.015, f"{v:.3f}", ha="center", va="top", fontsize=8, color="#ffffff")
    for xi, v in zip(x + width / 2, robustness["HB growth"]):
        ax.text(xi, v - 0.015, f"{v:.3f}", ha="center", va="top", fontsize=8, color="#ffffff")

    ax.set_xticks(x)
    ax.set_xticklabels([f"split {int(s)}\nn={int(n)}" for s, n in
                        zip(robustness["split"], robustness["n_common"])], fontsize=9)
    ax.set_ylabel("held-out log-likelihood per transition", color=INK_MUTED, fontsize=10)
    ax.set_ylim(min(robustness["HB growth"].min(), robustness["HMM"].min()) - 0.06, 0)
    diff = robustness["HMM"] - robustness["HB growth"]
    ax.set_title(f"Same held-out transitions: HMM ahead in {int((diff>0).sum())} of {len(diff)} splits "
                 f"(mean {diff.mean():+.4f})",
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.text(0.008, -0.07,
             "Higher (less negative) is better. Restricted to transitions where every visit has a finite ratio, so both "
             "models\ncan score them. Both are fitted only on training patients within each split.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)
