#!/bin/bash
# -----------------------------------------------------------------------------
# Submit one experiment to SLURM with a fresh, unique run directory.
#
#   scripts/cluster/submit.sh <run-name> [sbatch options --] <script.py> [args...]
#
# Examples
#   scripts/cluster/submit.sh bench -- scripts/cluster/benchmark_workloads.py --out @RUN_DIR@/results
#   scripts/cluster/submit.sh mogp_lopo --partition=gpu03 --time=2-00:00:00 -- \
#       scripts/21_mogp_heldout_comparison.py --out @RUN_DIR@
#
# The run directory is experiments/<UTC timestamp>_<run-name>/; the literal
# token @RUN_DIR@ in the script arguments is replaced by it. The directory is
# created here, never reused: a second submission gets a new timestamp. SLURM
# stdout/stderr go to <run_dir>/slurm-<jobid>.{out,err}.
#
# Recovery: every long-running driver checkpoints inside its run directory and
# accepts --resume <run_dir>; resubmit with the same @RUN_DIR@ replaced by that
# path (RESUME_DIR=<dir> scripts/cluster/submit.sh ...) to continue it.
# -----------------------------------------------------------------------------
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n '4,9p' "$0" >&2; exit 2; }
name="$1"; shift
sbatch_opts=()
if [[ " $* " == *" -- "* ]]; then
  while [[ "$1" != "--" ]]; do sbatch_opts+=("$1"); shift; done
  shift
fi
[[ -f pyproject.toml && -d scripts/cluster ]] || { echo "run from the repo root" >&2; exit 2; }

if [[ -n "${RESUME_DIR:-}" ]]; then
  run_dir="$(cd "$RESUME_DIR" && pwd)"
  [[ -d "$run_dir" ]] || { echo "no such run dir $RESUME_DIR" >&2; exit 2; }
else
  run_dir="$PWD/experiments/$(date -u +%Y%m%dT%H%M%SZ)_${name}"
  mkdir -p "$(dirname "$run_dir")"
  mkdir "$run_dir"                       # fails if it exists: never reuse
fi

args=()
for a in "$@"; do args+=("${a//@RUN_DIR@/$run_dir}"); done

jid=$(sbatch --parsable --export=ALL,RUN_DIR="$run_dir" --job-name="ppmi_${name}" \
       --output="$run_dir/slurm-%j.out" --error="$run_dir/slurm-%j.err" \
       "${sbatch_opts[@]}" scripts/cluster/job.slurm "${args[@]}")
echo "$jid" >> "$run_dir/slurm_job_ids.txt"
echo "submitted job $jid -> $run_dir"
