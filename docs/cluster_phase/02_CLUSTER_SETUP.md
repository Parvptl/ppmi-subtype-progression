# Cluster phase — Stage 2: environment, execution strategy, job procedure

Measured 2026-10-14. Benchmark numbers are in the generated
[`02_BENCHMARK_SYNTHETIC.md`](02_BENCHMARK_SYNTHETIC.md) (rendered from
`experiments/20261014T091904Z_cluster_benchmark/results/*.csv`); this file
only interprets them.

## 1. Hardware and scheduler

| | |
|---|---|
| Login node | `master` — **no GPU**, shared; do not run experiments here |
| Scheduler | SLURM 22.05.2 |
| GPU node | `node003`, partitions `gpu03` (max **5 days**) and `normal`; **2× NVIDIA A30 24 GB**, driver 530.30.02, 64 logical / 32 physical CPUs, ~100 GB RAM |
| Other nodes | `node004` (`gpu04`, 1 day), `node001` (draining), `node002` |
| Allocation | every partition is `OverSubscribe=EXCLUSIVE`: a job gets the **whole node**, so a CPU job still owns all 64 logical CPUs |
| Filesystem | `/home` is shared (10 TB, 88% used); `/tmp` is **node-local** — compute nodes cannot see files written to the login node's `/tmp` |

## 2. Python environment

```bash
module load anaconda3/2022.10
conda create -y -p ~/envs/ppmi -c conda-forge --override-channels python=3.13 pip
~/envs/ppmi/bin/pip install numpy==2.2.4 scipy==1.15.2 pandas==2.2.3 \
    matplotlib==3.10.1 pyyaml==6.0.3 openpyxl seaborn pytest==9.1.1 scikit-learn==1.6.1
~/envs/ppmi/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
~/envs/ppmi/bin/pip install -e ".[dev]" --no-deps
```

numpy/scipy/pandas match the versions every audited Windows run used. Python is
3.13.16 (Windows: 3.13.7). PyTorch is installed **only** for the GPU benchmark
(`scripts/cluster/torch_mogp.py`); no scientific result depends on it.
scikit-learn is needed by `scripts/20_clusterability_audit.py` but was never
declared in `pyproject.toml`.

Tests: **365 passed** on `node004` before any new code; **392** after the
Stage 3/4 additions (`tests/test_heldout.py`, `tests/test_hmm_stability.py`).

## 3. What is CPU-bound and what a GPU could change

Every model in the repository is NumPy/SciPy on many **small** matrices: GP
covariances of at most 28×28 per patient, 2×2 to 5×5 HMM generators, 2×2/4×4
LMM blocks. Nothing is written in a GPU framework.

| Workload | Character | GPU candidate? |
|---|---|---|
| MOGP / GP marginal likelihood (`mogp.objective`, `gpr._objective`) | batched Cholesky of ~934 matrices ≤28×28, per optimiser step | **yes, measured** |
| HMM forward algorithm (`hmm._forward_loglik`) | `expm` per unique gap + 3×3/5×5 recursions; finite-difference gradient | no — tiny, sequential, scipy `expm` |
| Joint/time-varying LMM | Woodbury with 2×2/4×4 blocks | no — already < 1 s per fit |
| Bootstrap / metrics | weighted sums | no — milliseconds after the multinomial-weight rewrite |

Measured on 934 synthetic patients:

* **CPU concurrency**: 60 single-thread MOGP fits run concurrently at ≈26× the
  single-fit throughput; 32 at ≈22×. Throughput flattens past the 32 physical
  cores, as expected.
* **GPU**: a PyTorch port of the MOGP objective agrees with the audited NumPy
  code to ~1e-12 in value and ~1e-10 in gradient, and evaluates ≈30–35× faster
  per call on one A30 than single-thread NumPy. A whole K1/K3 fit is ≈34–41×
  faster.
* **But** an L-BFGS-B run with the GPU objective does not follow the identical
  path: on K3 it stopped 0.003 NLL away with parameters differing by up to 0.07
  along a flat direction.

### Decision

Use the **audited NumPy implementation on CPU, parallel over folds**, for every
confirmatory run. Reasons:

1. 26× from CPU concurrency already brings the K2/K3 held-out comparison that
   was skipped as "65+ hours" down to hours (to be confirmed by the real-data
   timing run below), within the 5-day `gpu03` limit.
2. Mixing backends inside one comparison would make "identical folds, identical
   model" false at the 1e-3 level; switching *every* arm would replace a working,
   tested implementation for a speed gain that is not needed.
3. The GPU port is kept as a validated fallback (`torch_mogp.py`, agreement
   check in `benchmark_workloads.py`) should real-data timings be far worse than
   synthetic ones. Synthetic K2 fits take 21 s against the 250–320 s reported on
   real data on the laptop, so this is checked before launching.

The GPU node is still the right place to run: `gpu03` has the longest time
limit and the job owns all 64 CPUs.

## 4. Job submission

All jobs go through two files:

* `scripts/cluster/submit.sh <name> [sbatch opts] -- <script.py> [args]`
  creates `experiments/<UTC timestamp>_<name>/` (refuses to reuse a directory),
  substitutes `@RUN_DIR@` in the arguments, submits, and writes SLURM stdout/err
  into the run directory.
* `scripts/cluster/job.slurm` pins BLAS to one thread per process, refuses to
  run without `RUN_DIR`, and records the host, job id, git commit, `git diff`,
  a tarball of untracked source files, `pip freeze`, `nvidia-smi`, start/end
  time and exit status.

Example:

```bash
scripts/cluster/submit.sh s3_subscore --partition=gpu03 --time=2-00:00:00 -- \
    scripts/21_subscore_heldout_comparison.py --out @RUN_DIR@ --jobs 60
```

Monitor with `squeue -u $USER`, `tail -f experiments/<run>/slurm-<id>.out`,
`sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`.

## 5. Logging, checkpointing, recovery

* Unit of work = one (arm, held-out patient) task (Stage 3) or one
  (configuration, outer fold, inner fold) task (Stage 4). Each writes its own
  file atomically (`.tmp` + `os.replace`), the prediction file last, so a file's
  existence means the task finished.
* A killed or timed-out job is resumed by resubmitting the same command with
  `--resume` against the same run directory:

  ```bash
  RESUME_DIR=experiments/<run> scripts/cluster/submit.sh s3_subscore -- \
      scripts/21_subscore_heldout_comparison.py --out @RUN_DIR@ --resume
  ```

  The driver refuses to resume if the arms, seed, jitter, restarts, max-iter or
  panel MD5 differ from the original `run_config.json`, and appends each resume
  to `resume_log.jsonl`.
* A task that raises does not stop the run: its traceback goes to
  `failed_tasks.jsonl` and the job exits non-zero. Rerunning with `--resume`
  retries only failed and missing tasks.
* Every driver refuses to run on a panel whose MD5 is not the frozen
  `fb038ddbdc19ab3368e13940f398e7ba`, except with an explicit `--smoke` flag,
  which is recorded in the run metadata and stamped on the generated report.

## 6. Smoke tests

`experiments/_smoke/` holds a 120-patient synthetic panel with the real schema
and a config pointing at it. Scripts 21–24 have been run end to end on it,
including a deliberate resume after failures; the leakage battery passes.
These runs are not results.
