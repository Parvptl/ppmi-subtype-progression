"""Plot the Stage 5 period slopes with patient-bootstrap intervals (saved results only).

Reads experiments/20261014T100107Z_s5_temporal_audit/bootstrap/period_slopes_boot_vs_wald.csv
(primary model: knots at 2 and 5 years, random intercepts, 1,000 patient-bootstrap refits)
and writes report/figures/period_slopes.png. No model is fitted here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(__file__).resolve().parents[2]
src = R / "experiments/20261014T100107Z_s5_temporal_audit/bootstrap/period_slopes_boot_vs_wald.csv"
d = pd.read_csv(src).query("model == 'primary_RI'")
order = ["0-2y", "2-5y", "5-10y"]
labels = ["0-2 years", "2-5 years", "5+ years"]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
for i, (outcome, colour, name) in enumerate([("tremor", "#3b6ea8", "Tremor score"),
                                             ("pigd", "#c0504d", "PIGD score")]):
    g = d[d.outcome == outcome].set_index("period").loc[order]
    x = [k + (i - 0.5) * 0.18 for k in range(3)]
    ax.errorbar(x, g.slope, yerr=[g.slope - g.boot_lo, g.boot_hi - g.slope], fmt="o",
                color=colour, capsize=4, lw=1.6, label=name)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xticks(range(3), labels)
ax.set_xlabel("Period of follow-up")
ax.set_ylabel("Average change per year\n(score units)")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = R / "report/figures/period_slopes.png"
fig.savefig(out)
print("wrote", out)
