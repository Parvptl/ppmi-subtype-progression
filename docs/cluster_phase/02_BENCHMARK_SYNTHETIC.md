# Cluster benchmark — generated from `experiments/20261014T091904Z_cluster_benchmark/results`

Synthetic cohort with the real cohort's shape; no PPMI data read.

## Environment

```
{
  "host": "node003",
  "python": "3.13.16",
  "platform": "Linux-4.18.0-425.13.1.el8_7.x86_64-x86_64-with-glibc2.28",
  "numpy": "2.2.4",
  "cpu_count": 64,
  "slurm_job_id": "13301",
  "slurm_cpus_on_node": "64",
  "scipy": "1.15.2",
  "torch": "2.5.1+cu121",
  "cuda_available": true,
  "gpus": [
    "NVIDIA A30",
    "NVIDIA A30"
  ]
}
```

## One fit, one thread (934 synthetic patients)

| workload | seconds | converged |
|---|---|---|
| gp1_K1 | 0.2462 | True |
| gp1_K3 | 0.8661 | True |
| mogp_K0_full | 6.875 | True |
| mogp_K1_full | 9.202 | True |
| mogp_K1_diag | 4.346 | True |
| mogp_K2_full | 20.8 | True |
| mogp_K3_full | 25.52 | True |
| mogp_K3_diag | 7.922 | True |
| joint_lmm | 0.6363 | True |
| hmm3 | 1.494 | True |
| hmm5 | 39.75 | False |

## Concurrent single-thread fits on one node

| workload | processes | wall_sec | mean_fit_sec | fits_per_hour | speedup_vs_1 |
|---|---|---|---|---|---|
| mogp_K1_full | 1 | 9.943 | 9.146 | 362.1 | 1 |
| mogp_K1_full | 16 | 12 | 10.62 | 4798 | 13.25 |
| mogp_K1_full | 32 | 14.74 | 12.42 | 7815 | 21.58 |
| mogp_K1_full | 60 | 22.99 | 20.09 | 9396 | 25.95 |
| hmm3 | 1 | 2.281 | 1.498 | 1578 | 1 |
| hmm3 | 16 | 2.408 | 1.526 | 2.392e+04 | 15.15 |
| hmm3 | 32 | 3.705 | 1.902 | 3.109e+04 | 19.7 |
| hmm3 | 60 | 4.533 | 2.73 | 4.766e+04 | 30.2 |

## Objective evaluation: NumPy vs PyTorch (value + gradient)

`abs_value_diff` / `max_rel_grad_diff` are against the audited NumPy objective at the same parameter vector.

| kernel | coregion | backend | sec_per_eval | abs_value_diff | max_rel_grad_diff | speedup_vs_numpy |
|---|---|---|---|---|---|---|
| K1 | full | numpy_1thread | 0.1527 | 0 | 0 | 1 |
| K1 | full | torch_cpu | 0.08537 | 1.819e-12 | 6.616e-11 | 1.789 |
| K1 | full | torch_cuda | 0.004469 | 1.819e-12 | 6.616e-11 | 34.17 |
| K1 | diag | numpy_1thread | 0.1284 | 0 | 0 | 1 |
| K1 | diag | torch_cpu | 0.08768 | 2.728e-12 | 1.732e-11 | 1.465 |
| K1 | diag | torch_cuda | 0.004448 | 9.095e-13 | 1.732e-11 | 28.87 |
| K3 | full | numpy_1thread | 0.1625 | 0 | 0 | 1 |
| K3 | full | torch_cpu | 0.09413 | 9.095e-13 | 1.269e-11 | 1.726 |
| K3 | full | torch_cuda | 0.004718 | 1.819e-12 | 1.269e-11 | 34.43 |
| K3 | diag | numpy_1thread | 0.1369 | 0 | 0 | 1 |
| K3 | diag | torch_cpu | 0.09109 | 9.095e-13 | 1.931e-11 | 1.503 |
| K3 | diag | torch_cuda | 0.004696 | 9.095e-13 | 1.931e-11 | 29.16 |

## One full L-BFGS-B fit (restart 0): NumPy CPU vs PyTorch GPU

| kernel | numpy_sec | numpy_nfev | numpy_nll | gpu_sec | gpu_nfev | gpu_nll | nll_diff | max_abs_theta_diff | speedup |
|---|---|---|---|---|---|---|---|---|---|
| K1 | 8.579 | 61 | -7311 | 0.2526 | 57 | -7311 | 1.465e-05 | 0.0002289 | 33.97 |
| K3 | 23.84 | 145 | -7315 | 0.5841 | 128 | -7315 | 0.002628 | 0.06797 | 40.82 |

