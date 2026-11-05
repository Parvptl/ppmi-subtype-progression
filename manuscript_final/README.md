# manuscript_final

The complete, full-length manuscript reconstructed from the local repository on 5 November 2026, with its research audit, completeness report, figures, tables and build scripts. Everything in this directory was generated from, or copied from, the saved outputs of the experiments under `experiments/` and `outputs/`; nothing outside this directory was changed, no model was refitted and no job was submitted.

## Deliverables

| Path | What it is | Format |
|---|---|---|
| `full_research_paper.md` | The complete manuscript (title, authors, affiliation, abstract, keywords, introduction, objectives, related work, dataset, preprocessing, methods, design, evaluation protocol, 23 results subsections, discussion, limitations, conclusion, 26 references, supplementary S1-S4), with every table and figure embedded at its point of use | Markdown, about 52,000 words |
| `full_research_paper.pdf` | The same manuscript rendered with reportlab (`build/build_pdf.py`), 117 pages, A4 | PDF |
| `build/sections/*.md` | The editable source: one Markdown file per section, with `{{table:N}}` and `{{figure:N}}` placeholders. This is the same Markdown-section workflow the repository's `scripts/81_assemble_manuscript.py` and `scripts/86_build_manuscript_pdf.py` use; edit these, not the assembled file | Markdown |
| `research_audit.md` | Inspection record, the 40-experiment inventory with inclusion decisions, the exclusion audit for the clustering method review, numerical traceability, figure and table provenance, unresolved issues | Markdown |
| `manuscript_completeness.md` | Element-by-element completeness check, coverage of the requested investigation areas, verification status, what is deliberately absent and why | Markdown |
| `figures/` | The 70 figures used, copied unchanged from their source runs; `FIGURE_INDEX.csv` gives each one's source path, SHA-256 and caption | PNG, CSV |
| `tables/` | The 104 manuscript tables as machine-readable CSV (`tableNNN_<slug>.csv`), `TABLE_INDEX.csv` with captions and source files, and `numerical_traceability.csv` (1,077 checked numbers, all passing) | CSV |
| `build/asset_manifest.json` | Figure manifest (source path and caption for each figure) | JSON |
| `build/collect_assets.py` | Copies the figures and writes `figures/FIGURE_INDEX.csv` | Python |
| `build/make_tables.py` | Generates every table from the result files (rounding, interval formatting and, where the caption says so, a mean over saved per-split values are the only arithmetic) | Python |
| `build/assemble.py` | Assembles `full_research_paper.md` from `build/sections/` and the two index files | Python |
| `build/verify_numbers.py` | Checks every number quoted in the text against its source file and confirms the excluded material is absent; writes `tables/numerical_traceability.csv` | Python |
| `build/build_pdf.py` | Renders the assembled Markdown to PDF with reportlab (no pandoc or LaTeX needed) | Python |

## Rebuilding

From the repository root, with the project's environment (`~/envs/ppmi/bin/python`):

```bash
python manuscript_final/build/collect_assets.py && python manuscript_final/build/make_tables.py && python manuscript_final/build/assemble.py && python manuscript_final/build/verify_numbers.py && python manuscript_final/build/build_pdf.py
```

The scripts read only `experiments/`, `outputs/`, `docs/`, `manuscript/`, `paper/`, `report/` and `research_audit/`, and write only inside `manuscript_final/`. Fonts: DejaVu (system) for the PDF; it falls back to Helvetica if DejaVu is absent.

## Inputs and provenance

- Frozen visit panel `outputs/visit_panel.csv`, MD5 `fb038ddbdc19ab3368e13940f398e7ba`.
- Run directories listed in Table A of the manuscript (§7.4) and in `research_audit.md` §2; every number traces to a CSV, JSON or text file in one of them (`tables/numerical_traceability.csv`).
- References: the 26 entries carry the DOIs recorded in the project's Crossref-verified lists (`paper/ieee/refs.bib`, `docs/manuscript/03_*`); no citation was invented.

## Remaining issues (details in `research_audit.md` §6)

1. The author block follows the writing instruction (three authors, one affiliation); the repository's recent drafts carry four authors and two affiliations. The authors must confirm which is correct.
2. The PPMI acknowledgement and data-use statement, and the journal's required declarations (contributions, competing interests, funding, ethics, data and code availability), are not in the repository and were not invented.
3. The test suite collects 841 tests; the last recorded full run (758 passing) is from 4 November 2026 and the suite was not re-run for the manuscript.
4. Sixteen of twenty rich-profile fits stopped at the iteration limit; the five-state HMM converged in 404 of 934 folds; both are stated in the text and do not affect any conclusion.
5. Two specified pieces of work (the Stage 2 integer-score model; the M1 replicate-label diagnostic) were never run and are documented as such.
6. Figures are the working figures of their runs, not restyled to a journal format.
