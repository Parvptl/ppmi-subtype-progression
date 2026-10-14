# Cluster phase — experiment matrix and order

Each row is one run directory under `experiments/`, produced by one script,
analysed by one script that reads only saved files. **C** = confirmatory
(held-out, pre-specified comparisons). **E** = exploratory (in-sample or
descriptive; never reported as performance).

## Order, and why

0. **Gate.** Raw data present; panel rebuilt from raw into a separate run dir
   and its MD5 compared with the frozen `fb038ddb…`; the frozen panel used for
   every run. Full test suite on a compute node.
1. **Stage 3 first.** It closes two unsaved headline results (GP–LMM, GP–HMM)
   and the skipped K2/K3 held-out arms; its scoring module is reused later.
2. **Stage 4 next.** Independent of Stage 3, CPU-only, and its population stage
   tells us whether the HMM results everyone quotes rest on a unique optimum.
3. **Stage 5.** Uses the LMM machinery Stage 3 validated.
4. **Stage 6 last**, and only as an audit: it decides whether clustering is
   justified; it does not cluster unless the pre-specified rule says so.

## Stage 3 — `21_subscore_heldout_comparison.py` → `22_analyse_subscore_heldout.py`

| | |
|---|---|
| Cohort | 934 patients, 3,361 one-step points (identical to scripts 17/18 and the HMM LOO) |
| Folds | leave-one-patient-out; every arm refitted in every fold |
| Arms | persistence, patient_mean (fold-estimated error covariance), lmm_ri, gp_indep_K0–K3, gp_joint_K0–K3 × {diag, full} — 15 |
| Fit protocol | as audited: 1 restart from the deterministic start, max_iter 300, jitter 1e-8, seed 42 |
| Primary comparison (C) | nested cross-output: gp_joint_Kx_full − gp_joint_Kx_diag, x = K0..K3 |
| Secondary (C) | shared lengthscale (diag − indep), joint vs independent (full − indep), temporal flexibility (Kx − K0), every model − persistence, every GP − LMM |
| Continuous metrics | MAE, MSE/RMSE, CRPS, Gaussian log score, bivariate log score, coverage 50/80/90/95, PIT histogram + KS, negative predictive mass, bias |
| Subtype metrics | accuracy, balanced accuracy, per-class recall/precision, log loss, Brier, ECE, reliability tables; rules: point, prob-argmax, prob-prior-corrected (training prevalence) |
| Label persistence | script 28 pairs every derived subtype against repeating the last label (accuracy, balanced accuracy, Indeterminate recall) |
| HMM | script 09 LOO predictions joined on (patient, target index), validated on event ids and labels; paired subtype log loss and accuracy |
| Intervals | patient-cluster bootstrap, 4,000 replicates, seed 42, shared across every model and comparison |
| Checks | 11-item leakage/consistency battery; reproduction of the historical K0-full/K1-full/K1-diag/indep-K1/LMM predictions where those files exist |
| Population-only fits (E) | script 18 `kernel_parameters.csv` remains the only in-sample K2/K3 description; never mixed into held-out tables |

## Stage 4 — `23_hmm_stability.py` → `24_analyse_hmm_stability.py`

| | |
|---|---|
| Configurations | 3 states × floor {none, 0.5, 0.6, 0.7}; 4 states; 5 states; plus `C3_f50_audited` (= `hmm.fit_hmm`, 4 restarts, seed 42 — the original LOO protocol) |
| Population (E) | 40 dispersed starts per config on all 934 patients: optima, parameter spread at the optimum, termination messages, projected gradients |
| Held-out (C) | outer 10-fold patient CV, 10 dispersed starts per fit, best training likelihood kept; held-out score of **every** start recorded |
| Selection (C) | inner 5-fold CV log-likelihood on outer-training patients, and training BIC; selected config's outer predictions = unbiased estimate |
| Metrics | accuracy, balanced accuracy, per-class recall, log loss, Brier, ECE, Indeterminate AUC, mean P(Indet) by actual class; argmax vs prior-corrected rule |
| Reference | paired differences against `C3_f50_audited` on identical outer folds |

## Stage 5 — `25_temporal_audit.py`

| Stage | Type | What |
|---|---|---|
| describe | E | visits/patients/within-year pairs by follow-up year; retention by baseline PIGD tertile; late-period sparsity |
| bootstrap | C (inference) | 1,000 patient-bootstrap refits of the primary model (knots 2, 5; REML; RI and RS), percentile CIs vs Wald |
| knots | E + test | ML single-knot profile over 0.5–10 y; bootstrap argmax distribution; **simulated sup-LR test** against constant-rate nulls (RI and RS) |
| structures | E (fit) + C (CV) | linear, poly2, poly3, pw(3), pw(2,5), pw(1.5,4), pw(2.5,6), natural splines df 3–5 by ML AIC/BIC; period-average slopes with bootstrap CIs; 10-fold CV one-step prediction paired against linear |
| timeaxis | E | follow-up spline with and without disease-duration adjustment, if a duration column exists in the panel |
| attrition | E | loss of labelled follow-up vs current scores (logit, patient bootstrap); completers ≥ 6 y and ≥ 8 y; early slopes by eventual follow-up length |

## Stage 6 — `26_clusterability_extended.py`

| | |
|---|---|
| Features | BLUP random slopes (tremor, PIGD) from the random-slope LMM, refitted on every dataset |
| Nulls | 200 × Gaussian one-population; 200 × measurement-realistic (clipped, rounded to the item grid) |
| Positive control | planted two-group PIGD-slope differences of 0.5, 1, 2 random-slope SDs, 50 each |
| Perturbations | drop 1 / 2 random visits; truncate follow-up at 2, 3, 5 y |
| Rule (pre-specified) | supported only if real ΔBIC > null_measured q95 **and** drop-one ARI(K=2) > null_measured q95 and ≥ 0.6 |
| Restricted subset | ≥ 8 visits, same procedure, labelled exploratory |
