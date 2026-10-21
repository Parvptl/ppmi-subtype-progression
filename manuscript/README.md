# Manuscript: two methods for next-state prediction in PPMI

Research-meeting manuscript covering the two most recent methods in this
project:

1. **Probabilistic CT-HMM subtype prediction** — a continuous-time hidden
   Markov model of the motor subtype, predicting the next observed
   TD/Indeterminate/PIGD label from the label history.
2. **Rich longitudinal clinical profile** — per-domain levels and rates of
   change across nine clinical measurements, with levodopa dose as an input,
   predicting the next visit's measurements. Inspired by Severson et al.
   (2021); **not a reproduction of their architecture** (see §4.1 of the
   manuscript).

## Contents

| Path | What it is |
|---|---|
| `MANUSCRIPT.md` | the manuscript, source of truth, plain Markdown |
| `figures/*.png` | nine publication figures |
| `figures/*.csv` | the exact numbers drawn in each figure |
| `figures/figure_index.csv` | figure number, title, caption, companion data file |
| `figures/figure_manifest.json` | source runs, counts, example-patient rule, SHA-256 of every figure |
| `tables/table1_method_comparison.csv` | the head-to-head comparison table |
| `tables/table2..5_*.csv` | supporting tables copied verbatim from the runs |
| `tables/manuscript_number_check.csv` | every headline number re-read from source |
| `make_figures.py` | builds all figures and their companion CSVs |
| `make_tables.py` | builds the tables **and verifies the manuscript's numbers** |
| `render.py` | renders the manuscript to self-contained HTML |

## Regenerating everything

From the repository root, with the project environment:

```bash
~/envs/ppmi/bin/python manuscript/make_figures.py
~/envs/ppmi/bin/python manuscript/make_tables.py
~/envs/ppmi/bin/python manuscript/render.py
```

Takes well under a minute. Nothing is refitted: all three read the two frozen
experiment directories and write only into `manuscript/`.

## Source experiment runs

| Method | Run directory |
|---|---|
| 1 — CT-HMM | `experiments/20261020T104756Z_next_visit_td_pigd_prediction/` |
| 2 — Rich profile | `experiments/20261020T180515Z_clinical_profile_cv/` |

Both carry a `MANIFEST.sha256` over their own outputs and a `summary.json`
recording the environment, the seed and the automated checks (19/19 and 13/13
passed respectively). To point the scripts at different runs:

```bash
~/envs/ppmi/bin/python manuscript/make_figures.py --ct-hmm <dir> --profile <dir>
```

`experiments/` is git-ignored, since it contains participant-level data derived
from the PPMI panel and is not redistributable. The runs must therefore exist
locally; regenerate them with `scripts/60_next_visit_subtype_prediction.py` and
`scripts/64_clinical_profile_cv.py`.

## How the numbers are kept honest

`make_tables.py` re-reads every headline figure quoted in the manuscript out of
the run directories and compares it, at the precision printed, against a
hard-coded list of what the manuscript says. It exits non-zero on any
mismatch:

```
All 20 quoted numbers match their source files.
```

If the manuscript is edited, update `QUOTED` in `make_tables.py` and re-run.
A failure there means the prose and the experiments have drifted apart.

`make_figures.py` writes a companion CSV beside every figure holding exactly
the values plotted, so a figure can be audited without rerunning anything.

## A note on the PDF

The repository's historical route to PDF was: Markdown → HTML
(`scripts/_md_to_pdf.py`) → print with headless Chrome →
`scripts/_stamp_pages.py` for page numbers. **Chrome is not installed in this
environment**, and neither are weasyprint, wkhtmltopdf or pdflatex, so there is
no reliable automated PDF step here.

`render.py --pdf` will attempt one with PyMuPDF, but it checks the result and
**refuses to write it**: PyMuPDF's layout engine does not resolve the inlined
image URIs and paginates this document into 605 figureless pages. A broken PDF
is worse than none, because it looks like a deliverable.

To produce a PDF, open `MANUSCRIPT.html` in a browser and print to PDF. The
HTML is self-contained — the figures are embedded as base64 — so it needs no
accompanying files and the print CSS is already sized for A4.

`MANUSCRIPT.html` is **intentionally not committed**: it is a generated
artefact that duplicates the committed PNGs as base64 and is regenerable in one
command. The figures themselves are committed.

## Reading the manuscript

Start with the abstract and §6.1. The two results most worth discussing are:

- **Neither method is distinguishable from the other on the subtype endpoint,
  and neither clearly beats a persistence rule** — the label is highly
  persistent, so it is a weak endpoint for separating progression models.
- **The rich representation is better than the label as a *description* of a
  patient** — seven principal components for 90% of variance, and single
  coordinates beat the label by 0.11–0.16 AUC on prospective Hoehn & Yahr and
  MCI — **but the gain in prediction comes from the level, not the rate.**

§8 lists twelve limitations, including the differing validation protocols
between the two methods and the confounding-by-indication in the medication
coefficients.
