## 7. Experimental design

### 7.1 Validation designs

Four validation designs were used, and no number in this paper is compared across designs without saying so.

1. **A single stratified 70/30 patient split** (seed 42), with ten further random splits for robustness, for the first comparison of the chains and the CT-HMM (Section 9.5), the growth model's head-to-head (Section 9.7) and the first-visit model (Section 9.8). Held-out pairs on the main split: 1,002.
2. **Leave-one-patient-out (LOPO)**, 934 folds, for every three-class and two-class label model: the model is refitted from scratch on the other 933 patients and the held-out patient's labels are predicted one visit at a time from that patient's own earlier visits. Every fold refits from the same seed. The GPR runs use the same design on the 821 ratio-defined patients and the 934 subscore patients.
3. **5-fold patient cross-validation repeated four times** (20 fits) for the rich clinical profile and its baselines, whose Kalman-filter fits are too expensive for 934 refits; the derived-label comparison of the profile with the CT-HMM pairs the profile's cross-validated predictions with the CT-HMM's LOPO predictions on the same cases.
4. **Development and evaluation sets** for the latent clinical-state model: 280 development patients (four cells) for selecting the number of states, 654 evaluation patients under LOPO for the comparison.

### 7.2 Leakage control

Every validation driver carries explicit checks that are asserted at run time and written to a `validation_checks.csv` or `checks.json`: the held-out patient is absent from the training set; the history strictly precedes the target; the history length equals the target index; every arm is scored on identical points and observations; targets match the frozen panel; persistence equals the immediately preceding visit; standardisation statistics come from training patients only; no window used a padded or fabricated visit. The history-window experiments add the tamper test of Section 6.15 (zero spurious change over 934 folds and 11,644 cases). The ratio-band tail merging uses training observations only, and a held-out ratio above the training tail is still assignable.

### 7.3 Pre-specification and multiplicity

Each later experiment's write-up names its primary contrast before the run (labelled PRIMARY in the saved tables) and marks the rest secondary or exploratory; this paper keeps those labels. No correction for multiple comparisons was applied; instead every contrast is reported with its interval, the number of contrasts in each family is stated, and an isolated interval that excludes zero among many that do not is read as such. Exploratory groupings defined after prediction (the patient-level composition groups) are labelled exploratory.

### 7.4 Experiment inventory

Table A below lists every experiment included in this paper with its question, cohort, validation design, primary metric, run location and the section that reports it. The excluded and incomplete work is listed in Supplementary Section S3 and in the research audit that accompanies this manuscript.

| ID | Experiment | Question | Cohort (patients / units) | Validation | Primary metric | Run or output location | Status | Section |
|---|---|---|---|---|---|---|---|---|
| E01 | Cohort construction and participant flow | How many visits carry a label and why not? | 1,080 / 5,774 visits | descriptive | counts | `experiments/20261103T192414Z_e5_participant_flow/` | completed | 4.3 |
| E02 | Transition structure, reversion, conversion, interval dependence | How does the label move? | 934 / 3,361 pairs | patient bootstrap (2,000) | transition probabilities | `outputs/transitions/`; `experiments/20261103T193205Z_e2_transition_patient_bootstrap/` | completed | 9.1 |
| E03 | Markov-property test | Is the observed label first-order Markov? | 2,427 labelled triples | chi-square | chi-square | `outputs/model/markov_property_test.csv` | completed | 9.1 |
| E04 | Raw-score reconstruction and OFF/ON agreement (Part C) | Is the label reconstructed exactly, and how does it depend on the examination? | 1,080 / 5,774; 2,343 OFF/ON pairs | descriptive | mismatches; agreement | `experiments/20261015T221426Z_partC_raw_scores/` | completed | 9.2 |
| E05 | Indeterminate audit (Part C) | Is Indeterminate a transient boundary crossing? | 934 | patient bootstrap | reversion contrast, O/E clustering | `experiments/20261015T221426Z_partC_indeterminate/` | completed; pre-registered reading inconclusive | 9.2 |
| E06 | Threshold sensitivity (O-4) | How many labels move under a 0.02 threshold shift? | 1,080 / 4,441 | deterministic recount | share relabelled | `experiments/20261104T002308Z_o4_threshold_sensitivity/` | completed | 9.2 |
| E07 | Lag-1 residual autocorrelation (E-1) | Are within-patient residual changes independent? | 397 / 2,050 pairs | patient bootstrap, Monte Carlo null | lag-1 correlation | `experiments/20261103T191408Z_e1_lag1_residual_resolution/` | completed | 9.2 |
| E08 | Posterior-predictive label check (D-1) | Does the Student-t growth model reproduce the label dynamics? | 821 / 3,528 ratio visits | 5-fold, 500 replicates, 500 null cohorts | T = max abs z against c | `experiments/20261016T070746Z_d1_full/` | completed; not reproduced | 9.3 |
| E09 | D-1 Stage 1 diagnostics | Where does the misfit sit? | 821 / 3,528 | descriptive, quadrature | shares and rates | `experiments/20261016T133134Z_d1_stage1_checks/` | completed, no verdict | 9.3 |
| E10 | DTMC, CTMC and CT-HMM on the 70/30 split | Does a latent layer help? | 1,002 held-out pairs | 70/30 split, 10 further splits | held-out log-likelihood per pair | `outputs/model/` | completed | 9.4 |
| E11 | Leave-one-patient-out three-class prediction | Does the result hold under LOPO? | 934 / 3,361 | LOPO | accuracy, log-likelihood, Brier | `outputs/loo/` | completed | 9.5 |
| E12 | HMM stability study (Stage 4) | Is the fitted optimum unique and the choice of configuration consequential? | 934 / 3,361 | 40 starts; nested CV | log-likelihood spread; held-out accuracy | `experiments/20261014T094649Z_s4_hmm_stability/`; `experiments/20261014T100057Z_s4_hmm_cv/` | completed | 9.5 |
| E13 | Five-state HMM under LOPO | Do more latent states help? | 934 / 3,361 | LOPO | accuracy; convergence | `outputs/loo_5hidden/` | completed; converged in 404 of 934 folds | 9.5 |
| E14 | Hierarchical Bayesian growth model | Does a continuous growth model of the ratio predict the label? | 821 ratio-defined; 825 head-to-head pairs | 70/30 split; longitudinal CV | log-likelihood per pair | `outputs/hbayes/`; `outputs/validation/`; `outputs/exploration/` | completed | 9.6 |
| E15 | Student-t variants and Q2 reassessment | Are residuals or slopes heavy-tailed, and does it matter? | 821 | 7 patient splits; 4 chains | held-out log predictive density | `experiments/20261015T190228Z_student_t_four_model_full/`; `experiments/20261015T203325Z_q2_reassessment/` | completed | 9.6 |
| E16 | First-visit model | What can be predicted with no history? | 1,002 held-out rows | 10 splits | log-likelihood per prediction | `outputs/newpatient/` | completed | 9.7 |
| E17 | GPR of the ratio: kernels, mean function, multi-step | Does a flexible trajectory model help? | 821 / 2,707 one-step points | LOPO; 4 cohorts | MAE, CRPS | `outputs/step3_gpr/` | completed | 9.8 |
| E18 | Independent subscore GPs | Does modelling the subscores help? | 934 / 3,361 | LOPO | MAE | `outputs/step3_gpr/subscore_*`; `outputs/joint_subscore_gp/` | completed | 9.9 |
| E19 | Joint bivariate LMM | What is the interpretable baseline on the subscores? | 934 / 8,590 obs | LOPO | MAE; fixed effects | `outputs/joint_subscore/` | completed | 9.9 |
| E20 | Multi-output GP (LMC) | Does the cross-output block help? | 934 / 3,361 | LOPO | MAE, paired | `outputs/joint_subscore_gp/` | completed | 9.9 |
| E21 | Stage 3 held-out subscore comparison | Do K2/K3 and the joint covariance help on proper scores? | 934 / 3,361 | LOPO, 4,000 bootstrap | CRPS, log score | `experiments/20261014T094751Z_s3_subscore_heldout/analysis/` | completed | 9.9 |
| E22 | Time-varying linear-spline LMM | Does the population rate change over follow-up? | 934 | LOPO; Wald | period slopes | `outputs/time_varying/` | completed | 9.10 |
| E23 | Stage 5 temporal audit | Is the slope change identifiable and are the intervals honest? | 934 | parametric bootstrap, patient bootstrap | sup-LR; SE ratio | `experiments/20261014T100107Z_s5_temporal_audit/` | completed | 9.10 |
| E24 | Harmonised family comparison (E-8) | How do all families compare on one rule and one set? | 934 / 3,361 | re-scoring of saved LOPO predictions | accuracy, Brier, log loss | `experiments/20261103T202632Z_e8_harmonised_family_comparison/` | completed | 9.11 |
| E25 | Next-visit two-class prediction | Which of latent state and elapsed time matters for TD vs PIGD? | 899 / 3,029 | LOPO | accuracy, Brier, log loss | `experiments/20261020T104756Z_next_visit_td_pigd_prediction/` | completed | 9.12 |
| E26 | Latent clinical-state CT-HMM | Does a discrete latent clinical state beat the subtype history? | 280 dev; 654 eval / 2,091 | LOPO | accuracy | `experiments/20261020T142040Z_clinical_state_selection/`; `…142446Z_clinical_state_lopo_k6/`; `…145228Z_clinical_state_lopo_k8/`; `…145500Z_clinical_features_direct/` | completed; negative | 9.13 |
| E27 | Rich clinical profile (A-2) | Does a continuous level-and-rate profile predict the clinical record and carry outcome information? | 934 / 3,353; 30,099 domain obs | 5-fold CV × 4 | log-density per domain observation | `experiments/20261103T220147Z_a2p_outcome_table_fullprec/` | completed | 9.14 |
| E28 | Profile-derived two-class label (A-3) | Does the profile predict the next label better than the CT-HMM? | 898 / 3,020 | 5-fold CV vs LOPO, paired | accuracy | `experiments/20261104T085320Z_a3_profile_cv_with_baseline/` | completed | 9.14 |
| E29 | Model-selection criteria | What do AIC and BIC say within each likelihood family? | full cohort | in-sample | AIC, BIC | `experiments/20261023T194317Z_model_selection_criteria/` | completed | 9.15 |
| E30 | ROC and AUC | Does the profile rank better than the CT-HMM? | 898 / 3,020 | paired patient bootstrap | AUC | `experiments/20261023T202640Z_roc_auc_comparison/` | completed | 9.15 |
| E31 | Indeterminate boundary and focal-ratio scrutiny | What do the models say at Indeterminate visits? | 220 / 287 visits | patient bootstrap | agreement excess | `experiments/20261023T203431Z_indeterminate_boundary/`; `…211836Z_indeterminate_ratio_scrutiny/` | completed | 9.16 |
| E32 | Stage-wise evaluation | Does accuracy change with disease stage? | 896 / 3,017 | per-stage; within-patient | accuracy, balanced accuracy | `experiments/20261023T214819Z_stage_wise_evaluation/` | completed | 9.17 |
| E33 | CT-HMM history windows | How much history does the CT-HMM need? | 228 / 734 common targets | LOPO, paired | accuracy, AUC | `experiments/20261024T042417Z_history_window_experiment/` | completed | 9.18 |
| E34 | Profile history windows | How much history does the profile need? | 481 / 1,489 | 5-fold CV, paired | accuracy, AUC | `experiments/20261024T042811Z_profile_history_window/` | completed | 9.18 |
| E35 | CT-HMM retrained on two-visit windows | Does the training regime matter? | 742 / 2,192 | LOPO, paired | accuracy, balanced accuracy, AUC | `experiments/20261024T052343Z_retrained_two_visit_cthmm/` | completed | 9.19 |
| E36 | Experiment 6A: profile prediction history | Profile with last 1, 2, 3, 5 visits | 898 / 3,021 | 5-fold CV, paired | accuracy, AUC, PIGD sensitivity | `experiments/20261103T052237Z_profile_prediction_history/` | completed | 9.20 |
| E37 | Experiment 6B: profile two-visit retraining | Training by history 2 × 2 design | 740 / 2,184 | 5-fold CV, paired | accuracy, AUC | `experiments/20261103T072507Z_profile_two_visit_retraining/` | completed | 9.20 |
| E38 | Patient-level accuracy audit | How is accuracy distributed across patients? | 898 | per-patient, paired | per-patient accuracy | `experiments/20261103T175003Z_patient_level_accuracy_audit/` | completed | 9.21 |
| E39 | Ratio-banded CT-HMM, AUC and emission diagnostic | Does the ratio band sharpen the latent state? | 899 / 3,029 | LOPO, paired | accuracy; AUC | `experiments/20261104T152154Z_cthmm_ratio_banded_observations/`; `…172138Z_ratio_banded_auc_roc/`; `…155439Z_ratio_band_emission_diagnostic/` | completed | 9.22 |
| E40 | Band-width sweep | Is there a better band width? | 899 / 3,029 | LOPO, paired | AUC, PR-AUC, partial AUC | `experiments/20261105T003425Z_cthmm_ratio_band_width_sensitivity/`; `…025024Z_cthmm_ratio_band_width_extended/` | completed | 9.23 |

### 7.5 Compute

All fits ran on CPU nodes (the models are small and sequential, and the bootstrap, not the fit, dominates). The CT-HMM LOPO run takes minutes per fold-batch; the clinical profile's twenty fits are the largest single cost; the GPR mean-function hold-out took 44 minutes over 821 folds; the band-width sweep fitted 934 folds for each of six arms. No experiment was re-run for this manuscript; every number is read from the saved outputs of the runs listed above.
