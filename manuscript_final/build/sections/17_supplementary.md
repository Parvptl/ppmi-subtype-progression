## Supplementary material

### S1. Run identifiers and integrity

Every experiment reads the frozen visit panel `outputs/visit_panel.csv` (MD5 `fb038ddbdc19ab3368e13940f398e7ba`), verified by each run script. The run directories listed in Table A (Section 7.4) each hold the result tables this paper reads, the script's log, the git commit of the code, the installed package versions and, for runs from October 2026 onward, a SHA-256 manifest of the outputs. The threshold-sensitivity run wrote a separate perturbed panel (MD5 `a226105d178dc5dc38937a64f1dea9c4`) and left the frozen panel untouched. The D-1 run is frozen with lock checksums on its seven result files, and the Student-t runs record the SHA-256 of their specification and code.

The manuscript's own tables were generated from those result files by `manuscript_final/build/make_tables.py`, its figures were copied unchanged by `manuscript_final/build/collect_assets.py`, the text was assembled by `manuscript_final/build/assemble.py`, and every number quoted in the text was checked against its source file by `manuscript_final/build/verify_numbers.py`; the accompanying research audit lists the traceability table.

### S2. Software and tests

The pipeline is a Python 3.13.16 package (numpy 2.2.4, scipy 1.15.2, pandas 2.2.3; scikit-learn for the logistic models and the ranking metrics) run on CPU nodes of a SLURM cluster with one thread per fit. The test suite collects 841 tests covering the label rule and its degenerate cases, the panel construction, the transition and Kaplan-Meier estimators, each model's likelihood (the CT-HMM forward recursion against brute-force path enumeration, the banded model's reduction to the square model, the joint LMM's closed-form covariance against dense algebra, the Kalman filter, the Gaussian-emission marginalisation of missing features, the ratio-band arithmetic), the leakage checks of each validation driver, the Monte Carlo propagation through the classification rule, and the bootstrap helpers. The Stage 1 audit of 4 November 2026 recorded 758 tests passing at that date.

### S3. Work specified but not completed

Two pieces of work that the programme specified were not completed and are reported here so that the record is whole. Neither contributes a result to this paper.

1. **A measurement-faithful integer-score model (Stage 2).** The D-1 Stage 1 diagnostics (Section 9.3) recommended a model that generates the integer tremor and PIGD item sums directly, with no intermediate state, as the next test of whether the label's dynamics follow from its construction. The model was specified, implemented in several variants with adaptive-quadrature and reference likelihoods, and passed synthetic recovery tests (fixed-item and full-pipeline recovery), but its analysis of the real cohort was never run. Its run directories (`experiments/2026101[6-9]T*_stage2_*`) hold only pilots, benchmarks, diagnostics and recovery tests.
2. **The D-1 replicate-label diagnostic (M1).** A protocol-compliant re-execution of D-1's observed-cohort stage that would save the per-visit replicate labels, allowing the Stage 1 comparisons to be made against the patient-conditional reference D-1 actually used rather than the covariate-blind generator, was specified and reviewed for feasibility (about three minutes of compute on the original node) but not run, because it re-evaluates the observed cohort, which the frozen protocol allows only once with explicit approval.

### S4. Rebuilding this manuscript

From the repository root, with the project's Python environment:

```
python manuscript_final/build/collect_assets.py
python manuscript_final/build/make_tables.py
python manuscript_final/build/assemble.py
python manuscript_final/build/verify_numbers.py
python manuscript_final/build/build_pdf.py
```

The first two steps read only the saved experiment outputs; none of them fits a model or submits a job. The section sources under `manuscript_final/build/sections/` are the editable text; `full_research_paper.md` is assembled from them and should not be edited directly.
