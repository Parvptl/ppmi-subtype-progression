#!/bin/bash
# =============================================================================
#  Step 2 - patient-level leave-one-out for the 5-hidden-state HMM
#
#  20 array tasks, each holding out roughly 47 of the 934 patients. Sharding
#  divides the work, not the data: every fold still trains on all 933 other
#  patients, so the pooled result is identical to an unsharded run.
#
#  Submit from the project root:
#      mkdir -p logs/step2_5hidden
#      sbatch scripts/run_5hidden_loo.slurm
#
#  Then merge:
#      python scripts/merge_5hidden_loo.py
# =============================================================================
#SBATCH --job-name=ppmi_5hidden_loo
#SBATCH --partition=standard
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/step2_5hidden/shard_%a.out
#SBATCH --error=logs/step2_5hidden/shard_%a.err

set -euo pipefail

PROJECT_DIR=/home/interiit2025/parv_ppmi_hmm/project
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
#  Environment: the project's Conda environment, given by absolute path.
#
#  `conda activate` is preferred because it also runs the environment's own
#  activation scripts. If the conda shell hook is not available in a batch
#  shell, putting the environment's bin directory first achieves the same for a
#  prefix environment. Either way the interpreter is checked below, so the job
#  cannot quietly run against the wrong Python.
# ---------------------------------------------------------------------------
CONDA_ENV=/home/interiit2025/parv_ppmi_hmm/conda_env

if [ ! -x "$CONDA_ENV/bin/python" ]; then
    echo "ERROR: no Python at $CONDA_ENV/bin/python" >&2
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    export PATH="$CONDA_ENV/bin:$PATH"
    export CONDA_PREFIX="$CONDA_ENV"
fi

ACTIVE_PYTHON="$(command -v python || true)"
if [ "$ACTIVE_PYTHON" != "$CONDA_ENV/bin/python" ]; then
    echo "ERROR: expected $CONDA_ENV/bin/python but got '${ACTIVE_PYTHON:-none}'" >&2
    exit 1
fi

# Each task owns 4 cores. The matrices here are 5x5, so threaded BLAS buys
# nothing and oversubscription across 20 concurrent tasks would cost time.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUM_SHARDS=20

# PPMI_CONFIG must not leak in from a Step 1 shell: Step 2 uses the original
# thresholds. The shard script also refuses to start if they are wrong.
unset PPMI_CONFIG || true

echo "host            $(hostname)"
echo "job             ${SLURM_JOB_ID:-none}  array task ${SLURM_ARRAY_TASK_ID:-none}"
echo "shards          ${NUM_SHARDS}"
echo "python          $(which python)"
echo "started         $(date -Is)"
echo

python -u scripts/10_loo_validation_5hidden_shard.py

echo
echo "finished        $(date -Is)"
