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
| `make_pdf.py` | renders the manuscript to PDF via an installed browser |

## Regenerating everything

From the repository root, with the project environment:

```bash
~/envs/ppmi/bin/python manuscript/make_figures.py
~/envs/ppmi/bin/python manuscript/make_tables.py
~/envs/ppmi/bin/python manuscript/render.py
~/envs/ppmi/bin/python manuscript/make_pdf.py        # needs a browser
```

Takes well under a minute. Nothing is refitted: all four read the two frozen
experiment directories and write only into `manuscript/`.

The first two need the experiment runs below; `render.py` and `make_pdf.py` need
only `MANUSCRIPT.md` and `figures/`, so the document can be re-rendered on a
machine that does not hold the participant-level data.

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

The route to PDF is: Markdown → HTML (`scripts/_md_to_pdf.py`) → print with
headless Chrome → `scripts/_stamp_pages.py` for the running footer and page
numbers. `make_pdf.py` runs all three:

```bash
python manuscript/make_pdf.py
```

It needs Chrome, Edge or Brave on the machine (it looks in the usual install
locations; `--browser PATH` overrides) and it writes `MANUSCRIPT.pdf` here:
19 pages, A4, nine figures embedded at 222–341 dpi. On a machine with no
browser it exits saying so rather than writing something broken. It was Chrome's
absence that blocked this step on the cluster — weasyprint, wkhtmltopdf and
pdflatex were missing there too.

### Partial renders

`--through-section N` stops the document after numbered section N, leaving out
everything from N+1 onward (including trailing unnumbered sections such as
"Data and code availability") and trimming the separator rule left dangling at
the cut:

```bash
python manuscript/make_pdf.py --through-section 7 \
    --out manuscript/MANUSCRIPT_sections_1_7.pdf \
    --title "Next clinical state in Parkinson's disease - sections 1-7"
```

That is `MANUSCRIPT_sections_1_7.pdf` as it stands: 17 pages, §§1–7 with all
nine figures; §8 Limitations, §9 Conclusion, §10 References and "Data and code
availability" omitted. The body still carries its citations, so the excerpt is
for readers who do not need to follow them.

`--keep-section N` (repeatable) adds a section back after the cut, which is how
to restore the reference list if a later excerpt wants it:

```bash
python manuscript/make_pdf.py --through-section 7 --keep-section 10 ...
```

A kept section keeps its original number — the references come back as "§10"
rather than renumbered to follow §7, since renumbering is an edit to the
manuscript's structure rather than a rendering choice.

The excerpt is derived from `MANUSCRIPT.md` at render time rather than from a
second copy of the text, so the two documents cannot drift apart.

### Report style

`--style report` renders the document as a plain LaTeX-style article instead of
the project's blue screen styling: one serif face, no colour anywhere, justified
text with indented paragraphs, booktabs rules on the tables (no fill, no
vertical lines), the abstract set narrower, and a bare centred page number with
no running footer. It also drops the bold strapline under the title, which is a
manuscript-for-a-meeting line that a report does not want. The front matter is
supplied on the command line:

```bash
python manuscript/make_pdf.py --through-section 6 --style report \
    --authors "Parv Patel    ·    Mayank Tewatia" \
    --affiliation "Supervisor: Prof. K. R. Sahasranand" \
    --affiliation "Department of Data Science, IIT Palakkad" \
    --date "October 2026" \
    --out manuscript/MANUSCRIPT_sections_1_6.pdf
```

`--subtitle`, `--authors`, `--affiliation` (repeatable) and `--date` all default
to nothing, so with none of them the page carries only the title. None of this
comes from `MANUSCRIPT.md`, which keeps its own title block untouched.

The typeface is Latin Modern — the Computer Modern revival LaTeX sets by
default — taken from the local MiKTeX or TeX Live installation and staged beside
the HTML, since those files are not registered as system fonts. Without them it
falls back to Cambria and says so. Chrome embeds the staged font as outline
Type3; glyphs are vector and text stays extractable, but note that some
submission systems dislike Type3. Installing Latin Modern as a system font
avoids it.

### Why `make_pdf.py` exists alongside `render.py`

They differ in **one** thing: how the figures reach the document. `render.py`
inlines each PNG as a base64 `data:` URI, so the HTML is a single file that can
be e-mailed with no image directory. That is the right choice for the HTML and
the wrong one for printing — it makes a 2.8 MB document whose data URIs headless
Chrome will not finish laying out, exiting without writing a file. `make_pdf.py`
calls the same `embed_images` with `inline_data=False`, so the figures keep their
relative paths and are staged beside the HTML in a temp directory. Everything
else — the Markdown conversion, the stylesheet, the figure and caption styling —
is imported from the same two modules, so the PDF matches the HTML.

`make_pdf.py` adds only print-specific CSS: tables of ten rows or fewer are
pinned to a single page (longer ones may still split, with the header repeated,
which is `_md_to_pdf.py`'s deliberate choice), and figure captions cannot be
separated from their figure.

### A bug this uncovered

Figures used to be substituted into the Markdown as raw `<figure>` HTML, on the
assumption — stated in `render.py`'s docstring — that the project renderer passes
through block-level HTML it does not recognise. **It does not.**
`_md_to_pdf.convert` has no raw-HTML branch, so such a line fell through to its
paragraph handler, which runs `inline()` → `html.escape()` and turned every tag
into visible text. `MANUSCRIPT.html` therefore contained **no images at all** —
just nine escaped `&lt;figure&gt;` blocks and their base64 payloads as a wall of
characters. The "9 figures embedded" message counted substitutions attempted,
not images that survived.

This also corrects the old diagnosis recorded here. PyMuPDF's "605 figureless
pages" was not its layout engine failing to resolve image URIs; it was faithfully
typesetting megabytes of literal base64 text. Headless Chrome hung for the same
reason.

Figures are now spliced in **after** conversion, via a placeholder that survives
escaping (`render.splice_figures`), and an unspliced placeholder is a hard error
rather than a nonsense word in the output. Captions go through the project's
`inline()`, so the leading `**Figure n.**` renders bold instead of printing its
asterisks.

`render.py --pdf` remains, and still **refuses to write a file**: PyMuPDF's Story
engine paginates this document poorly and inflates the inlined images into a
file of tens of megabytes, so the result is checked and rejected. A broken PDF is
worse than none, because it looks like a deliverable. Use `make_pdf.py`, or open
`MANUSCRIPT.html` in a browser and print to PDF — the print CSS is already sized
for A4. (On Windows the rejected temp file can fail to delete, leaving a
`.tmp.pdf` to remove by hand.)

### Checking a rendered PDF

```bash
python scripts/_verify_manuscript_pdf.py
```

Re-reads `MANUSCRIPT.pdf` and reports anything missing against `MANUSCRIPT.md`
and `MANUSCRIPT.html` — headings, figure captions, embedded images and their
placed resolution, table rows, references, page-number footers — and fails on
stray markup such as a literal `**`, an unspliced placeholder or an escaped tag.
It exists because a PDF that silently drops its figures still opens and still
looks like a deliverable, which is exactly what the earlier route produced.

To check a partial render, give it the same `--through-section` and
`--keep-section` and it compares against the same truncation of the source (it
imports the rule from `make_pdf.py`, so the two cannot disagree about where the
cut falls):

```bash
python scripts/_verify_manuscript_pdf.py \
    --pdf manuscript/MANUSCRIPT_sections_1_7.pdf --through-section 7
```

For a `--style report` render add `--footer plain` (the page number is a bare
centred numeral, not "Page n of m") and `--no-strapline`, which asserts that the
line under the title really is gone:

```bash
python scripts/_verify_manuscript_pdf.py \
    --pdf manuscript/MANUSCRIPT_sections_1_6.pdf \
    --through-section 6 --footer plain --no-strapline
```

### What is and is not committed

`MANUSCRIPT.html` is **intentionally not committed**: it is a generated artefact
that duplicates the committed PNGs as base64 and is regenerable in one command.
The figures themselves are committed.

`MANUSCRIPT.pdf` is **not committed either**, though less deliberately: the
repository's `.gitignore` excludes `*.pdf` wholesale, excepting only `report/`,
`reports/` and `extra/`. Committing the manuscript PDF would need a
`!manuscript/*.pdf` exception.

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
