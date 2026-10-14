#!/bin/bash
# =============================================================================
#  Generic PPMI experiment job for the IIT Palakkad cluster (Rocky 8, SLURM 22).
#
#  Submit through scripts/cluster/submit.sh, which creates the run directory
#  and log directory BEFORE sbatch opens the log files, and refuses to reuse a
#  run directory. Direct use:
#
#      sbatch --export=ALL,RUN_DIR=<dir> scripts/cluster/job.slurm <python args...>
#
#  Partitions (all OverSubscribe=EXCLUSIVE: a job owns the whole node):
#      gpu03   node003  2x A30 24GB, 64 logical CPUs, ~100 GB   max 5 days
#      gpu04   node004  64 logical CPUs, ~100 GB                max 1 day
#      normal  node001-004                                      max 1 day
#  Because the node is exclusive, a CPU-only experiment still gets all 64
#  logical CPUs (32 physical); request what the script will actually use.
# =============================================================================
#SBATCH --job-name=ppmi
#SBATCH --partition=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00

set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set (use scripts/cluster/submit.sh)}"
PROJECT_DIR="${PPMI_PROJECT_DIR:-$HOME/projects/ppmi-subtype-progression}"
PY="${PPMI_PYTHON:-$HOME/envs/ppmi/bin/python}"
cd "$PROJECT_DIR"

[ -x "$PY" ] || { echo "ERROR: no interpreter at $PY" >&2; exit 2; }

# One BLAS thread per process: every workload here is many small matrices, so
# parallelism comes from worker processes, never from threaded BLAS.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
# Configs other than the baseline must be requested explicitly per run.
if [ -z "${PPMI_CONFIG_EXPLICIT:-}" ]; then unset PPMI_CONFIG || true; fi

{
  echo "host        $(hostname)"
  echo "job         ${SLURM_JOB_ID:-none}  partition ${SLURM_JOB_PARTITION:-?}"
  echo "cpus        ${SLURM_CPUS_ON_NODE:-?}   gpus ${CUDA_VISIBLE_DEVICES:-none}"
  echo "python      $PY"
  echo "git         $(git rev-parse HEAD)  dirty=$(git status --porcelain | wc -l)"
  echo "config      ${PPMI_CONFIG:-config/config.yaml}"
  echo "run_dir     $RUN_DIR"
  echo "started     $(date -Is)"
  echo "command     $PY $*"
} | tee "$RUN_DIR/job_header.txt"

git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git diff > "$RUN_DIR/git_diff.patch" || true
# Untracked source is invisible to `git diff`; archive it so the exact code
# that ran is recoverable even before it is committed.
git ls-files --others --exclude-standard -z -- src scripts tests config \
  | tar --null -czf "$RUN_DIR/untracked_source.tar.gz" -T - 2>/dev/null || true
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true
command -v nvidia-smi >/dev/null && nvidia-smi > "$RUN_DIR/nvidia_smi.txt" 2>&1 || true

set +e
"$PY" -u "$@"
status=$?
set -e
echo "finished    $(date -Is)  exit=$status" | tee -a "$RUN_DIR/job_header.txt"
exit $status
