# Manuscript completeness report

Date: 5 November 2026. Manuscript: `manuscript_final/full_research_paper.md` (about 52,000 words, 104 tables, 70 figures, 26 references) and its PDF rendering `full_research_paper.pdf`.

## 1. Required elements

| Element | Present | Where | Notes |
|---|---|---|---|
| Title | yes | line 1 | the supplied title, unchanged |
| Authors | yes | front matter | exactly as supplied: Parv Patel, Mayank Tewatia, and Dr. Sahasranand K. R. |
| Affiliation | yes | front matter | Indian Institute of Technology Palakkad, India |
| Abstract | yes | structured, about 560 words | background, methods, results, conclusions |
| Keywords | yes | after the abstract | ten keywords |
| Introduction | yes | §1 | |
| Research objectives and contributions | yes | §2 | six objectives, nine contributions |
| Related work | yes | §3 | 26 references, every one with a DOI from the project's verified lists; none invented |
| Dataset | yes | §4 | source, label rule, participant flow (Tables 1-3, Figure 1), descriptive characteristics, frozen panel |
| Preprocessing | yes | §5 | examination selection, item handling, time axis and covariates, software and tests |
| Methods (mathematical, matching the implementation) | yes | §6 | 15 subsections; every model written as the code implements it, with parameter counts, bounds, optimiser settings and the decision rules |
| Experimental design | yes | §7 | validation designs, leakage control, pre-specification, the full inventory (Table A, 40 experiments), compute |
| Evaluation protocol | yes | §8 | targets, metrics, uncertainty, reporting rules |
| Results, one subsection per experiment or linked group | yes | §9.1 to §9.23 | each with question, setup and cohort, baselines, results and uncertainty, figures and tables, interpretation, limitations and contribution |
| Figures | yes | 70, embedded at the point of use with captions and source paths | copies of existing run figures; none regenerated |
| Tables | yes | 104, embedded with captions and source paths | generated from result files by `build/make_tables.py`; machine-readable copies in `tables/` |
| Discussion | yes | §10 | mechanism, what richer observations buy, relation to the stability and progression literatures, methodological lessons, implications |
| Limitations | yes | §11 | twelve items |
| Conclusion | yes | §12 | |
| References | yes | numbered 1-26 | |
| Supplementary material | yes | S1-S4 | run identifiers and integrity, software and tests, work specified but not completed, how to rebuild |

## 2. Coverage of the investigation areas requested

| Area | Covered in | Status |
|---|---|---|
| Subtype definition and label stability | §4.2, §9.1, §9.2 | complete; transition structure, reversion, sustained conversion, interval dependence, Markov test, OFF/ON agreement, Indeterminate audit, threshold sensitivity, lag-1 autocorrelation |
| Dataset and cohort construction | §4, §5 | complete; participant-flow ledger with every exclusion counted |
| Observed-state Markov models (DTMC, CTMC) | §6.2, §9.4, §9.5, §9.11, §9.12 | complete |
| Hidden Markov models (3-state CT-HMM, 5-state) | §6.3, §9.4, §9.5 | complete; stability study and five-state convergence failure reported |
| Continuous-time modelling | §6.2, §6.3, §9.4, §9.12 | complete; the continuous-time chain loses to the discrete chain, and the reason is given |
| Hierarchical Bayesian growth model | §6.4, §9.6 | complete; identifiability of the two variance terms stated |
| Gaussian-process experiments (ratio, subscores, multi-output, kernels, mean function, multi-step) | §6.6, §6.7, §9.8, §9.9 | complete |
| Probabilistic TD-PIGD programme (CT-HMM, rich clinical profile, persistence, history windows, retraining, patient-level composition, Indeterminate handling, calibration, AUC, Brier, log loss) | §6.10 to §6.15, §9.12 to §9.23 | complete |
| History-window experiment | §9.18 to §9.21 | complete, for the CT-HMM and the profile |
| Ratio-based and ratio-banded experiments; band-width sensitivity | §6.13, §9.22, §9.23 | complete; the band-width sweep was completed and is included |
| Additional completed experiments (D-1 posterior-predictive check and Stage 1 diagnostics; Student-t variants and the Q2 reassessment; Stage 3, 4 and 5 analyses; first-visit model; model-selection criteria; Indeterminate boundary scrutiny; stage-wise evaluation) | §9.3, §9.5, §9.6, §9.7, §9.9, §9.10, §9.15, §9.16, §9.17 | complete |

## 3. Verification status

- Every number quoted in the running text was checked against its source file by `build/verify_numbers.py`: 1,077 of 1,077 checks passed on 5 November 2026 (`tables/numerical_traceability.csv`).
- Every table was generated from the result files named in `tables/TABLE_INDEX.csv`; every figure was copied unchanged from the file named in `figures/FIGURE_INDEX.csv` with its SHA-256 recorded.
- The assembled manuscript contains none of the strings that would indicate the excluded review (automated check in `build/verify_numbers.py`), and the exclusion is recorded in `research_audit.md` §3.
- The author block and affiliation match the instruction exactly; no contributions, e-mail addresses, ORCID identifiers or corresponding-author details were added.
- No original experiment file, output, report or configuration was modified, moved, deleted or regenerated; no job was submitted; nothing was committed or pushed; no PPMI data are reproduced.

## 4. What the manuscript does not contain, and why

| Item | Reason |
|---|---|
| The clustering method review and everything derived from it | excluded by the authors' instruction; recorded in `research_audit.md` §3 |
| The Stage 2 integer-score measurement model | incomplete: synthetic recovery tests only, never run on the real cohort (Supplementary S3) |
| The M1 replicate-label diagnostic | planned and reviewed, never run (Supplementary S3) |
| The docstring claim of a growing CT-HMM advantage with history (+0.005 to +0.047) | no saved result file behind it (§8.4) |
| The stale closing sentence of `outputs/exploration/report.txt` on residual structure | contradicted by E-1 (§9.2) |
| Author contributions, competing interests, funding, ethics and consent statements, data- and code-availability statements, the PPMI acknowledgement wording | not in the repository; must be supplied by the authors (`research_audit.md` §6) |
| Journal-specific formatting, restyled figures, a LaTeX source | the Markdown source and its PDF are the deliverables; the repository's IEEE LaTeX workflow is a separate, shorter paper |

## 5. Remaining issues before submission

See `research_audit.md` §6: the author-block discrepancy with the repository's recent drafts; the PPMI acknowledgement and data-use statement; the required declarations; the test-count statement; the rich-profile iteration-limit note; the five-state HMM convergence note; the validation-design difference between the profile and the CT-HMM; the two incomplete pieces of work; a check of references 25 and 26 against the journal's style; figure restyling.
