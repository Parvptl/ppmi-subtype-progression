# IEEE manuscript

A journal paper in IEEEtran format, written from this repository's code and
its committed results. Two-column `\documentclass[journal]{IEEEtran}`. No
journal is named in the source: the running header carries the short title, so
the same file suits whichever venue you choose, and a conference layout needs
only the class options and the `\markboth` line changed.

**Title.** Towards Longitudinal Prediction of Parkinson's Disease Motor
Subtypes.

This is a separate manuscript from `manuscript/npj/`. Nothing here reads or
writes that directory.

## Files

| File | What it is |
|---|---|
| `main.tex` | The manuscript. The only file written by hand. |
| `refs.bib` | 17 references, every one checked against Crossref, PubMed and OpenAlex on 2026-11-04 and resolving to a real record. |
| `make_assets.py` | Builds `figures/`, `tables/` and `derived_numbers.json` from `outputs/` and the frozen `experiments/` runs. Contains no analysis logic. |
| `verify_numbers.py` | Rebuilds every quantitative claim in `main.tex` from `derived_numbers.json` and asserts it appears in the text. 91 checks. |
| `build.sh` | assets → PDF → number check. |
| `figures/*.pdf` | Five figures, generated. |
| `tables/*.tex` | Six table bodies, generated; `main.tex` `\input`s them. |
| `derived_numbers.json` | Every number the text quotes, keyed, with its full precision. |
| `numbers_check.csv` | Output of the last verification run. |
| `main.pdf` | Compiled, 12 pages. |

## Rebuilding

```bash
./paper/ieee/build.sh
```

Needs `~/envs/ppmi/bin/python` (pandas, numpy, matplotlib) and a LaTeX
toolchain. `tectonic` at `~/envs/tex/bin/tectonic` is used if present and
downloads what it needs, including `IEEEtran.cls` and `IEEEtran.bst`;
`latexmk` is the fallback. Override either with `PPMI_PYTHON` or `TECTONIC`.

To work on it in Overleaf instead, upload this whole directory and set
`main.tex` as the root document. Everything it needs is inside.

## Where the numbers come from

Nothing in this directory recomputes a result. `make_assets.py` reads:

| Source | Supplies |
|---|---|
| `outputs/visit_panel.csv` | Cohort characteristics (Table I), the Markov-property conditional probabilities |
| `outputs/transitions/` | Interval-stratified transitions, Kaplan–Meier curves |
| `outputs/loo/`, `outputs/loo_5hidden/` | Three-class metrics, calibration, per-class recall, five-state convergence |
| `outputs/model/` | Markov-property test, emission matrix, mean sojourn times |
| `experiments/…_e2_transition_patient_bootstrap/` | Transition probabilities and every reversion/survival interval (Table II, Fig. 2) |
| `experiments/…_e5_participant_flow/` | Participant flow and the unlabelled-visit diagnosis (Fig. 1) |
| `experiments/…_e8_harmonised_family_comparison/` | The harmonised 14-specification table and the margin decomposition (Tables III–IV, Figs. 3–4) |
| `experiments/…_o4_threshold_sensitivity/` | Threshold perturbation and attainable ratio values (Table V, Fig. 4b) |
| `experiments/…_model_selection_criteria/` | Two-class matched evaluation, AIC/BIC (Table IV) |
| `experiments/…_roc_auc_comparison/` | Two-class AUCs and their paired contrast |
| `experiments/…_history_window_experiment/`, `…_retrained_two_visit_cthmm/` | History-window and retraining contrasts |
| `experiments/…_stage_wise_evaluation/` | Stage-wise performance |
| `experiments/…_patient_level_accuracy_audit/` | Per-participant distribution and composition gradient |
| `experiments/…_a2p_outcome_table_fullprec/` | Representation dimensionality and outcome discrimination (Table VI, Fig. 5). This is the full-precision rerun; the earlier `…_a1_outcome_contrast_orientation/` run computed the same contrasts from a `.round(4)` copy and its own script marks them superseded |

Tables, figure labels and the running text all round half away from zero
(`fmt()` in `make_assets.py`, `_round()` in `verify_numbers.py`), so a value
printed in a table and the same value quoted in a sentence always agree.

## Still needed from the authors

The manuscript carries two visible `[To be completed before submission]`
markers, for things that exist nowhere in the repository:

1. **PPMI acknowledgement** — the full funding-partner list, verbatim from the
   current PPMI publication policy, is required by the Data Use Agreement.
2. **Declarations** — ethics/consent wording for the target journal, competing
   interests, funding, corresponding author and contact details, ORCIDs, and
   author contributions.

Also unset: the journal's own submission metadata (manuscript type, suggested
reviewers, highlights or a graphical abstract if the venue asks for one).
