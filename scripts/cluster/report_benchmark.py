"""Render a benchmark run's saved CSVs as Markdown. No number is typed by hand.

    python scripts/cluster/report_benchmark.py <run_dir>/results <out.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def md(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(res_dir: str, out: str) -> int:
    res = Path(res_dir)
    env = json.loads((res / "env.json").read_text())
    parts = [f"# Cluster benchmark — generated from `{res}`\n",
             "Synthetic cohort with the real cohort's shape; no PPMI data read.\n",
             "## Environment\n", "```", json.dumps(env, indent=2), "```\n"]

    single = pd.read_csv(res / "single_fit_times.csv")
    parts += ["## One fit, one thread (934 synthetic patients)\n", md(single), ""]

    if (res / "scaling.csv").exists():
        sc = pd.read_csv(res / "scaling.csv")
        base = sc[sc.processes == 1].set_index("workload").fits_per_hour
        sc["speedup_vs_1"] = sc.apply(lambda r: r.fits_per_hour / base[r.workload], axis=1)
        parts += ["## Concurrent single-thread fits on one node\n", md(sc), ""]

    if (res / "gpu_vs_cpu.csv").exists():
        g = pd.read_csv(res / "gpu_vs_cpu.csv")
        ref = g[g.backend == "numpy_1thread"].set_index(["kernel", "coregion"]).sec_per_eval
        g["speedup_vs_numpy"] = g.apply(
            lambda r: ref[(r.kernel, r.coregion)] / r.sec_per_eval, axis=1)
        parts += ["## Objective evaluation: NumPy vs PyTorch (value + gradient)\n",
                  "`abs_value_diff` / `max_rel_grad_diff` are against the audited NumPy "
                  "objective at the same parameter vector.\n", md(g), ""]
    if (res / "gpu_full_fit.csv").exists():
        f = pd.read_csv(res / "gpu_full_fit.csv")
        f["speedup"] = f.numpy_sec / f.gpu_sec
        parts += ["## One full L-BFGS-B fit (restart 0): NumPy CPU vs PyTorch GPU\n",
                  md(f), ""]
    Path(out).write_text("\n".join(parts) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
