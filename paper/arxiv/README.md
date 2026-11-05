# arXiv preprint source

**Towards Longitudinal Prediction of Parkinson's Disease Motor Subtypes**
Parv Patel, Mayank Tewatia, Varsha Vasudevan, K. R. Sahasranand

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Produces `main.pdf`. Requires `newtx`, `natbib`, `authblk`, `booktabs`,
`geometry`, `caption`, `fancyhdr`, `microtype`, `hyperref`.

## Contents

| Path | Contents |
|---|---|
| `main.tex` | Manuscript source |
| `refs.bib` | Bibliography (17 entries, each resolved against Crossref, PubMed and OpenAlex) |
| `figures/` | Figures 1–7, copied unmodified from the committed experiment outputs |
| `tables/` | Table bodies 1–5, included by `main.tex` |
| `verify_numbers.py` | Re-derives every headline number in `main.tex` from the committed result files |

## Number verification

```bash
python -I paper/arxiv/verify_numbers.py
```

Run from the repository root. 90 checks, all passing as of 5 November 2026.
The matched Gaussian-process-versus-hidden-Markov-model contrast is recomputed
from the two saved prediction files rather than read from a stored summary.

## Figure provenance

Figures are copied without modification from the committed outputs; no
numerical value, interval or class definition was altered.

| Figure | Source |
|---|---|
| 1 Participant flow | `manuscript_final/figures/fig01_participant_flow.png` |
| 2 Label instability | `manuscript_final/figures/fig02_label_instability.png` |
| 3 Harmonised comparison | `manuscript_final/figures/fig34_harmonised_family_comparison.png` |
| 4 Margin decomposition | `manuscript_final/figures/fig35_margin_decomposition.png` |
| 5 Attainable ratios | `manuscript_final/figures/fig13_attainable_ratios.png` |
| 6 Growth model | `manuscript_final/figures/fig27_growth_trajectories.png` |
| 7 Ratio GP comparison | `manuscript_final/figures/fig29_gpr_model_comparison.png` |

## Outstanding before posting

The PPMI Data Use Agreement requires the current list of PPMI funding partners
to be reproduced verbatim in the acknowledgement. That list is not recorded in
this repository and has deliberately not been invented; it must be inserted
from the current PPMI publication policy before the preprint is posted.
