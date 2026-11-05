# arXiv preprint — full programme

**Towards Longitudinal Prediction of Parkinson's Disease Motor Subtypes**
Parv Patel, Mayank Tewatia, Varsha Vasudevan, K. R. Sahasranand

A comprehensive preprint covering the whole research programme: the main paper
plus supplementary material in one PDF (73 pages).

This is **separate from** `paper/arxiv/`, which is the shorter preprint built
around the four model families of the conference abstract. Neither overwrites
the other, and `manuscript_final/` is untouched.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires `newtx`, `natbib`, `authblk`, `booktabs`, `longtable`, `geometry`,
`caption`, `fancyhdr`, `microtype`, `hyperref`.

## Contents

| Path | Contents |
|---|---|
| `main.tex` | Main manuscript (Sections 1–11) |
| `supplementary.tex` | Supplementary material (S1–S15), `\input` by `main.tex` |
| `refs.bib` | 26 references, each resolved against Crossref, PubMed and OpenAlex |
| `figures/` | 16 main figures (`f01`–`f16`), 11 supplementary (`s01`–`s11`) |
| `tables/` | 13 main table bodies (`t01`–`t13`), 12 supplementary (`s01`–`s12`) |
| `verify_numbers.py` | Re-derives 93 headline numbers from the committed result files |

## Number verification

```bash
python -I paper/preprint/verify_numbers.py
```

Run from the repository root. 93 checks, all passing as of 6 November 2026. The
matched Gaussian-process-against-hidden-Markov-model contrast is recomputed from
the two saved prediction files rather than read from a stored summary, because
the project's own audit records that contrast as never having been saved.

## Figure provenance

Figures are byte copies of the committed experiment outputs, with one exception:
three figures carried an internal figure or experiment identifier baked into
their title bar, and that title band was cropped away with `ffmpeg` so the
identifier does not appear in the paper. No numerical value, interval, axis,
class definition or colour mapping was altered in any figure.

| Cropped | Source | Reason |
|---|---|---|
| `f02_lattice.png` | `manuscript_final/figures/fig09_label_lattice.png` | title carried an internal figure ID |
| `f16_patient.png` | `manuscript_final/figures/fig56_patient_accuracy_full_history.png` | title carried an internal experiment ID |
| `s02_exit_ind.png` | `manuscript_final/figures/fig12_time_to_exit_indeterminate.png` | title carried an internal figure ID |

## Experiment completeness checklist

Every completed analysis in the programme is accounted for. The paper's own
inventory is Table S14; this table adds the internal identifiers used elsewhere
in the repository, which deliberately do **not** appear in the manuscript.

| Internal ID | Analysis | Placement |
|---|---|---|
| E01 | Cohort construction and participant flow | main §3 |
| E02 | Transition structure, reversion, conversion, interval | main §8.1 |
| E03 | Markov-property test | main §8.1 |
| E04 | Raw-score reconstruction; OFF/ON agreement | main §4 |
| E05 | Pre-registered Indeterminate audit | main §8.1, supp S1 |
| E06 | Threshold sensitivity | main §4 |
| E07 | Lag-1 residual autocorrelation | main §8.1 |
| E08 | Posterior-predictive label check | main §8.2 |
| E09 | Diagnostic localisation of the misfit | main §8.2, supp S2 |
| E10 | Chains vs latent-state model, first split | supp S3 |
| E11 | Leave-one-patient-out three-class | main §8.3 |
| E12 | Optimum uniqueness, configuration sensitivity | supp S3 |
| E13 | Five-state model (negative) | main §8.3, supp S3 |
| E14 | Hierarchical Bayesian growth model | main §8.4 |
| E15 | Student-t variants | supp S5 |
| E16 | First-visit model | supp S4 |
| E17 | Gaussian-process regression of the ratio | main §8.4 |
| E18 | Independent subscore Gaussian processes | main §8.4 |
| E19 | Joint bivariate mixed model | supp S6 |
| E20 | Multi-output Gaussian processes | supp S6 |
| E21 | Fifteen-arm held-out subscore comparison | supp S6 |
| E22 | Time-varying spline model | supp S7 |
| E23 | Temporal identifiability and interval audit | supp S7 |
| E24 | Harmonised family comparison + margin decomposition | main §8.5 |
| E25 | Two-class next-visit prediction | main §8.6 |
| E26 | Discrete latent clinical states (negative) | main §8.6, supp S8 |
| E27 | Continuous nine-domain profile | main §8.6 |
| E28 | Profile-derived label | main §8.6 |
| E29 | Information criteria | supp S9 |
| E30 | Threshold-free discrimination (AUC) | main §8.6 |
| E31 | Model calls at Indeterminate visits | supp S9 |
| E32 | Stage-wise evaluation | supp S10 |
| E33 | Latent-state model history windows | main §8.7, supp S11 |
| E34 | Profile history windows | supp S11 |
| E35 | Retraining on two-visit windows (negative) | supp S11 |
| E36 | Profile with truncated history | supp S11 |
| E37 | Profile two-visit retraining | supp S11 |
| E38 | Patient-level accuracy audit | main §8.7, supp S12 |
| E39 | Ratio-banded observation alphabet | main §8.6 |
| E40 | Band-width sweep | main §8.6, supp S13 |
| X02 | Integer-score measurement model | supp S15 — **specified, never run on the cohort** |
| X03 | Re-execution saving replicate labels | supp S15 — **specified, never run** |
| X01 | Clustering method review | **excluded by author instruction** — contributes to nothing in this paper |

The excluded clustering line of work is referred to in the manuscript only as
"a separate line of work on unsupervised structure in the cohort … outside the
scope of this paper" (Section S14), with no finding, figure, table or number
derived from it.

## PPMI acknowledgement

The Acknowledgements section carries both statements the PPMI Data Use Agreement
requires: the data-source statement with RRID:SCR_006431, and the funding-partner
list reproduced verbatim from the current PPMI publication policy.

The data-source statement gives the acquisition date as **2026-08-20**. That is
the date `docs/DATA_REQUIREMENTS.md` records for the two MDS-UPDRS item files,
which were the last component of the analysed dataset to be obtained; the
curated cut and participant-status files came on 2026-07-26. If PPMI expects the
date of first access rather than last, change it in `main.tex` under
`\section*{Acknowledgements}`.
