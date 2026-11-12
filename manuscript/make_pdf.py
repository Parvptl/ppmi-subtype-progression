"""Render MANUSCRIPT.md to a print-quality PDF using an installed browser.

This is the repository's historical route, which ``render.py`` documents as
unavailable: Markdown -> HTML -> headless Chrome ``--print-to-pdf`` ->
``scripts/_stamp_pages.py`` for the running footer and page numbers. On this
machine Chrome *is* installed, so the route works.

Why this exists alongside ``render.py``
--------------------------------------
``render.py`` inlines every figure as a base64 ``data:`` URI, because its goal
is a single self-contained file that can be e-mailed with no image directory.
That is the right choice for the HTML and the wrong one for printing: the nine
PNGs make a 2.8 MB document whose data URIs headless Chrome will not finish
laying out (it exits without writing a file, even with a long
``--virtual-time-budget``).

So this script changes exactly one thing about the document: figures become
ordinary relative ``<img src="figures/...">`` references, staged next to a copy
of the PNGs. Everything else -- the Markdown conversion, the base stylesheet and
the figure/caption styling -- is imported from ``scripts/_md_to_pdf.py`` and
``render.py`` and used verbatim, so the PDF matches the committed HTML.
``MANUSCRIPT.md``, the figures and both existing renderers are untouched.

Staging also sidesteps a second problem: the repository lives under a path
containing a space, which Chrome's ``--print-to-pdf`` handles unreliably. The
staging directory is created under the system temp root, which has none.

    python manuscript/make_pdf.py
"""

from __future__ import annotations

import argparse
import html as html_mod
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

#: Where a Windows install of Chrome or Edge normally lives. Either renders this
#: document identically; both use the same Blink print path.
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
]

#: Additions to the project stylesheet that matter only on paper. The base CSS
#: in ``scripts/_md_to_pdf.py`` already repeats table headers, keeps rows whole
#: and forbids a break between a heading and what follows it; these rules close
#: the remaining cases where a page break lands badly.
PRINT_CSS = """
/* A figure and its caption are one unit; never separate them. render.py's
   FIGURE_CSS already says this, repeated here so it survives either import. */
figure, figcaption { break-inside: avoid; page-break-inside: avoid; }
figure { break-before: auto; }
figcaption { break-before: avoid; page-break-before: avoid; }

/* Do not leave a heading, or a heading plus one line, stranded at a page foot. */
h1, h2, h3, h4 { break-after: avoid-page; page-break-after: avoid; }
h2, h3 { break-inside: avoid; page-break-inside: avoid; }

/* Keep a table's header row attached to at least some body rows rather than
   letting a break fall immediately after the header. */
thead { break-after: avoid; page-break-after: avoid; }
tbody tr:first-child { break-before: avoid; page-break-before: avoid; }

/* Short tables are kept whole (see keep_short_tables_together); long ones may
   still split with the header repeated, which is the project's deliberate
   choice -- forbidding it outright leaves large blank gaps (_md_to_pdf.py). */
table { break-inside: auto; }
table.keep-together { break-inside: avoid; page-break-inside: avoid; }

p { orphans: 3; widows: 3; }
img { break-inside: avoid; page-break-inside: avoid; }
"""


def load_module(path: Path, name: str):
    """Import a module by file path (both targets have leading-underscore or
    script-style names that are not importable normally)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: Matches a numbered top-level section, e.g. "## 8. Limitations".
SECTION = re.compile(r"^##\s+(\d+)\.\s", re.M)


def _trim_trailing_rules(text: str) -> str:
    """Drop horizontal rules left dangling at the end of a slice.

    Sections are separated by a rule, so cutting at a heading strands the rule
    that introduced it.
    """
    text = text.rstrip()
    while True:
        head, _, last = text.rpartition("\n")
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", last.strip()):
            text = head.rstrip()
        else:
            return text


def extract_section(md: str, number: int) -> str | None:
    """Return numbered top-level section ``number``, heading included."""
    start = None
    for m in SECTION.finditer(md):
        if int(m.group(1)) == number:
            start = m.start()
            break
    if start is None:
        return None
    nxt = re.search(r"^##\s", md[start + 1:], re.M)
    end = start + 1 + nxt.start() if nxt else len(md)
    return _trim_trailing_rules(md[start:end])


def truncate_sections(md: str, through: int,
                      keep: tuple[int, ...] = ()) -> tuple[str, list[str]]:
    """Keep the manuscript up to the end of section ``through``.

    Cuts at the first numbered top-level heading above ``through``; anything
    after it goes, including trailing unnumbered sections such as "Data and
    code availability". Sections listed in ``keep`` are then appended back in
    ascending order, which is how an excerpt retains the reference list that
    its body still cites. Returns (markdown, titles_dropped).

    Kept sections keep their original numbering: renumbering §10 to follow §7
    would be an edit to the manuscript's structure, not a rendering choice.

    Deriving the short version here rather than keeping a second copy of
    MANUSCRIPT.md means the two documents cannot drift apart: there is still
    exactly one source of truth for the text.
    """
    cut = None
    for m in SECTION.finditer(md):
        if int(m.group(1)) > through:
            cut = m.start()
            break
    if cut is None:
        return md, []

    body = _trim_trailing_rules(md[:cut])
    kept_titles = []
    for n in sorted(set(keep)):
        section = extract_section(md, n)
        if section is None:
            raise SystemExit(f"--keep-section {n}: no such numbered section")
        kept_titles.append(section.splitlines()[0].lstrip("# ").strip())
        body += "\n\n---\n\n" + section

    dropped = [t for t in (l.lstrip("# ").strip()
                           for l in md[cut:].splitlines()
                           if l.startswith("## "))
               if t not in kept_titles]
    return body + "\n", dropped


def figures_referenced(md: str, base: Path) -> list[str]:
    """Relative paths of every figure the manuscript references and that exists.

    Used to decide what to copy into the staging directory; whether each one
    actually reaches the document is ``render.embed_images``' business.
    """
    render = load_module(HERE / "render.py", "ms_render")
    found = []
    for line in md.splitlines():
        m = render.IMAGE.match(line.strip())
        if m and (base / m.group("path")).resolve().exists():
            found.append(m.group("path"))
    return found


#: Latin Modern, the Computer Modern revival LaTeX sets by default. Shipped by
#: MiKTeX/TeX Live but not registered with Windows, so the files are staged
#: beside the HTML and pulled in with @font-face.
LM_DIRS = [
    Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/lm",
    Path(r"C:\Program Files\MiKTeX\fonts\opentype\public\lm"),
    Path("/usr/share/texmf/fonts/opentype/public/lm"),
    Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm"),
]
LM_FACES = {
    "lmroman10-regular.otf": ("normal", "normal"),
    "lmroman10-bold.otf": ("bold", "normal"),
    "lmroman10-italic.otf": ("normal", "italic"),
    "lmroman10-bolditalic.otf": ("bold", "italic"),
}


def find_latin_modern() -> Path | None:
    for d in LM_DIRS:
        if (d / "lmroman10-regular.otf").exists():
            return d
    return None


def latin_modern_css(available: bool) -> str:
    """@font-face block for the staged Latin Modern files, or nothing."""
    if not available:
        return ""
    return "".join(
        f"@font-face{{font-family:'Latin Modern Roman';"
        f"src:url('fonts/{name}') format('opentype');"
        f"font-weight:{weight};font-style:{style};}}"
        for name, (weight, style) in LM_FACES.items())


#: A plain report stylesheet in the shape of a LaTeX ``article``: one serif
#: face, no colour, justified text with indented paragraphs, booktabs rules on
#: the tables and a bare centred page number. Replaces the project's screen
#: stylesheet wholesale rather than overriding it piecemeal.
REPORT_CSS = """
@page { size: A4; margin: 24mm 24mm 26mm 24mm; }
* { box-sizing: border-box; }
body {
  font-family: "Latin Modern Roman", "Latin Modern Roman 10", Cambria,
               "Times New Roman", Georgia, serif;
  font-size: 10.5pt; line-height: 1.34; color: #000;
  margin: 0; padding: 0; text-align: justify;
  hyphens: auto; -webkit-hyphens: auto;
}

/* Title block ---------------------------------------------------------- */
.titleblock { text-align: center; margin: 6pt 0 20pt; }
.titleblock h1 {
  font-size: 17.5pt; font-weight: bold; line-height: 1.25;
  margin: 0 auto 4pt; padding: 0; border: none; color: #000;
  page-break-before: avoid; text-align: center; max-width: 92%;
}
.titleblock .subtitle { font-size: 11pt; margin: 8pt 0 0; color: #000; }
.titleblock .authors { font-size: 11.5pt; margin: 14pt 0 0; }
.titleblock .affil { font-size: 9.6pt; margin: 3pt 0 0; }
.titleblock .date { font-size: 10.5pt; margin: 13pt 0 0; }

/* Abstract: narrower measure and slightly smaller, as in an article ----- */
.abstract { margin: 4pt 7% 18pt; font-size: 9.7pt; }
.abstract h2 {
  text-align: center; font-size: 10.5pt; font-weight: bold;
  border: none; margin: 0 0 6pt; padding: 0; color: #000;
}
.abstract p { text-indent: 0; margin: 0 0 4pt; }
.abstract p + p { text-indent: 1.4em; }

/* Headings ------------------------------------------------------------- */
h1, h2, h3, h4 {
  color: #000; border: none; padding: 0; font-weight: bold;
  text-align: left; page-break-after: avoid; break-after: avoid-page;
  line-height: 1.25;
}
h2 { font-size: 13pt; margin: 17pt 0 7pt; }
h3 { font-size: 11.4pt; margin: 13pt 0 5pt; }
h4 { font-size: 10.5pt; margin: 11pt 0 4pt; }

/* Body: LaTeX-style indented paragraphs, no blank line between them ----- */
p { margin: 0 0 0; text-indent: 1.4em; orphans: 3; widows: 3; }
h1 + p, h2 + p, h3 + p, h4 + p, .titleblock + p,
figure + p, table + p, ul + p, ol + p, blockquote + p { text-indent: 0; }
strong { font-weight: bold; color: #000; }
em { font-style: italic; }
a { color: #000; text-decoration: none; }

ul, ol { margin: 5pt 0 6pt; padding-left: 22pt; text-indent: 0; }
li { margin-bottom: 2pt; }
li p { text-indent: 0; }

code {
  font-family: "Latin Modern Mono", Consolas, "Courier New", monospace;
  font-size: 9.3pt; background: none; padding: 0; color: #000;
}
pre {
  font-family: "Latin Modern Mono", Consolas, "Courier New", monospace;
  font-size: 9pt; background: none; border: none; border-left: 1pt solid #888;
  padding: 2pt 0 2pt 9pt; margin: 7pt 0; line-height: 1.3;
  white-space: pre-wrap; word-break: break-word; text-indent: 0;
  page-break-inside: avoid;
}
blockquote {
  margin: 7pt 5%; padding: 0; background: none; border: none;
  font-size: 10pt; text-indent: 0; page-break-inside: avoid;
}
blockquote p { text-indent: 0; margin: 0; }

/* The source separates sections with a rule; an article does not. */
hr { display: none; }

/* Tables: booktabs -- horizontal rules only, no fill, no vertical lines - */
table {
  border-collapse: collapse; margin: 10pt auto 11pt; font-size: 9.3pt;
  max-width: 100%; text-align: left; text-indent: 0;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td {
  border: none; padding: 3.1pt 7pt; vertical-align: top; text-align: left;
}
thead th {
  font-weight: bold; background: none; color: #000;
  border-top: 1.1pt solid #000; border-bottom: 0.5pt solid #000;
}
tbody tr:last-child td { border-bottom: 1.1pt solid #000; }
tbody tr:nth-child(even) { background: none; }
td.num, th.num { text-align: right; white-space: nowrap; }
table.keep-together { break-inside: avoid; page-break-inside: avoid; }

/* Figures -------------------------------------------------------------- */
figure {
  margin: 12pt 0 14pt; text-align: center;
  break-inside: avoid; page-break-inside: avoid;
}
figure img { max-width: 100%; height: auto; border: none; }
figcaption {
  font-size: 9.1pt; text-align: justify; margin-top: 6pt; line-height: 1.33;
  text-indent: 0; color: #000;
}
figcaption strong, figcaption b { font-weight: bold; color: #000; }
.missing-figure {
  border: 1px dashed #000; padding: 8pt; font-size: 9pt; text-align: left;
}
"""

#: A table of at most this many body rows is forced to stay on one page. Chosen
#: so the biggest protected table is about a third of a page: large enough to
#: cover every table in this manuscript bar none, small enough that pushing one
#: to the next page cannot leave an unsightly gap.
KEEP_TOGETHER_MAX_ROWS = 10


def keep_short_tables_together(html_text: str) -> tuple[str, int, int]:
    """Mark tables short enough to fit on one page. Returns (html, kept, split).

    Chrome will otherwise break a table wherever the page ends, which on this
    document stranded two or three rows of a five- and an eight-row table at the
    top of the following page. CSS alone cannot express "only if it is short",
    so the row count is measured here and a class added.
    """
    kept = split = 0

    def mark(m: re.Match) -> str:
        nonlocal kept, split
        table = m.group(0)
        body = re.search(r"<tbody>(.*?)</tbody>", table, re.S)
        rows = len(re.findall(r"<tr>", body.group(1))) if body else 0
        if rows and rows <= KEEP_TOGETHER_MAX_ROWS:
            kept += 1
            return table.replace("<table>", '<table class="keep-together">', 1)
        split += 1
        return table

    return re.sub(r"<table>.*?</table>", mark, html_text, flags=re.S), kept, split


def strip_source_subtitle(md: str) -> tuple[str, str | None]:
    """Drop the standalone bold line under the title, and the rule after it.

    The manuscript carries a meeting strapline there. A report does not want
    it, and MANUSCRIPT.md is not ours to edit, so it is removed at render time.
    Returns (markdown, removed_line).
    """
    lines = md.splitlines()
    i = next((k for k, l in enumerate(lines) if l.startswith("# ")), None)
    if i is None:
        return md, None
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines) or not re.fullmatch(r"\*\*.+\*\*", lines[j].strip()):
        return md, None
    removed = lines[j].strip()
    del lines[j]
    while j < len(lines) and not lines[j].strip():
        del lines[j]
    if j < len(lines) and re.fullmatch(r"-{3,}", lines[j].strip()):
        del lines[j]
    return "\n".join(lines), removed


def build_title_block(html_text: str, subtitle: str | None,
                      authors: str | None, affiliations: list[str],
                      date: str | None) -> str:
    """Wrap the <h1> in a centred title block and add the report front matter."""
    m = re.search(r"<h1>(.*?)</h1>", html_text, re.S)
    if not m:
        return html_text
    parts = [f"<h1>{m.group(1)}</h1>"]
    if subtitle:
        parts.append(f'<div class="subtitle">{html_mod.escape(subtitle)}</div>')
    if authors:
        parts.append(f'<div class="authors">{html_mod.escape(authors)}</div>')
    for line in affiliations:
        parts.append(f'<div class="affil">{html_mod.escape(line)}</div>')
    if date:
        parts.append(f'<div class="date">{html_mod.escape(date)}</div>')
    block = '<div class="titleblock">' + "".join(parts) + "</div>"
    return html_text[:m.start()] + block + html_text[m.end():]


def wrap_abstract(html_text: str) -> str:
    """Put the Abstract in its own narrower block, as an article does."""
    m = re.search(r"<h2>\s*Abstract\s*</h2>", html_text, re.I)
    if not m:
        return html_text
    nxt = re.search(r"<h[12][ >]", html_text[m.end():])
    end = m.end() + (nxt.start() if nxt else 0)
    if not nxt:
        return html_text
    return (html_text[:m.start()] + '<div class="abstract">'
            + html_text[m.start():end] + "</div>" + html_text[end:])


def stamp_plain_page_numbers(path: Path) -> int:
    """Centred page number at the foot of each page, and nothing else.

    ``scripts/_stamp_pages.py`` draws a running title and "Page n of m", which
    other reports in this repository use; a plain report wants only the number,
    so this is done here rather than by changing that shared script.
    """
    import pymupdf

    doc = pymupdf.open(path)
    total = doc.page_count
    for n, page in enumerate(doc, start=1):
        label = str(n)
        width = pymupdf.get_text_length(label, fontname="tiro", fontsize=10)
        page.insert_text(
            pymupdf.Point((page.rect.width - width) / 2, page.rect.height - 42),
            label, fontname="tiro", fontsize=10, color=(0, 0, 0))
    tmp = path.with_suffix(".stamped.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(path)
    return total


def find_browser(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"no browser at {explicit}")
        return explicit
    for cand in BROWSERS:
        if Path(cand).exists():
            return cand
    raise SystemExit(
        "No Chrome/Edge/Brave found. Pass --browser PATH, or install one.")


def print_to_pdf(browser: str, html: Path, pdf: Path, profile: Path,
                 timeout: int) -> None:
    """Drive headless Chrome's print path.

    Deliberately *without* --virtual-time-budget: on a document this size that
    flag makes Chrome hang and exit having written nothing. Local <img> files
    are loaded before first paint, so the budget buys nothing here anyway.
    """
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf}",
        str(html),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if not pdf.exists():
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("the browser exited without writing a PDF")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(HERE / "MANUSCRIPT.md"))
    ap.add_argument("--out", default=str(HERE / "MANUSCRIPT.pdf"))
    ap.add_argument("--browser", default=None)
    ap.add_argument("--title", default="Next clinical state in Parkinson's disease")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--through-section", type=int, default=None, metavar="N",
                    help="stop after numbered section N (e.g. 7); everything "
                         "from section N+1 onward is left out")
    ap.add_argument("--keep-section", type=int, action="append", default=[],
                    metavar="N",
                    help="add numbered section N back after the cut (e.g. 10 "
                         "for the references); repeatable")
    ap.add_argument("--style", choices=("default", "report"), default="default",
                    help="'report': plain LaTeX-article look -- serif, no "
                         "colour, centred page number, no running footer")
    ap.add_argument("--subtitle", default=None,
                    help="report style: line under the title (default none)")
    ap.add_argument("--authors", default=None, help="report style: author line")
    ap.add_argument("--affiliation", action="append", default=[],
                    help="report style: a line under the authors; repeatable")
    ap.add_argument("--date", default=None, help="report style: date line")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    browser = find_browser(args.browser)

    raw = src.read_text(encoding="utf-8")
    if args.through_section is not None:
        raw, dropped = truncate_sections(raw, args.through_section,
                                         tuple(args.keep_section))
        kept = ", ".join(str(n) for n in sorted(set(args.keep_section)))
        print(f"stopping after section {args.through_section}"
              + (f", keeping section {kept}" if kept else "")
              + f"; left out {len(dropped)} section(s):")
        for d in dropped:
            print(f"    {d}")
    elif args.keep_section:
        raise SystemExit("--keep-section only makes sense with "
                         "--through-section")
    used = figures_referenced(raw, src.parent)

    md_to_pdf = load_module(ROOT / "scripts" / "_md_to_pdf.py", "_md_to_pdf")
    render = load_module(HERE / "render.py", "ms_render")

    # inline_data=False: figures keep their relative paths and are staged beside
    # the HTML. Splicing happens after conversion, since the project renderer
    # escapes raw HTML rather than passing it through.
    md, blocks, missing = render.embed_images(raw, src.parent,
                                             inline_data=False)
    report = args.style == "report"
    lm_dir = find_latin_modern() if report else None
    if report:
        md, removed = strip_source_subtitle(md)
        if removed:
            print(f"report style: dropped the strapline {removed[:60]}...")
        if lm_dir is None:
            print("  Latin Modern not found; falling back to Cambria/Times")

    body, n_spliced = render.splice_figures(md_to_pdf.convert(md), blocks)
    body, kept, splittable = keep_short_tables_together(body)
    if report:
        body = build_title_block(body, args.subtitle, args.authors,
                                 args.affiliation, args.date)
        body = wrap_abstract(body)
        style = latin_modern_css(lm_dir is not None) + REPORT_CSS
    else:
        style = f"{md_to_pdf.CSS}{render.FIGURE_CSS}{PRINT_CSS}"
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html_mod.escape(src.stem)}</title>"
        f"<style>{style}</style>"
        f"</head><body>{body}</body></html>")

    # Staged under the temp root: no spaces in the path, and the figures sit
    # beside the HTML so the relative src values resolve.
    stage = Path(tempfile.mkdtemp(prefix="ms_pdf_"))
    try:
        (stage / "manuscript.html").write_text(document, encoding="utf-8")
        for rel in sorted(set(used)):
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src.parent / rel, dst)
        if lm_dir is not None:
            (stage / "fonts").mkdir(exist_ok=True)
            for name in LM_FACES:
                shutil.copy2(lm_dir / name, stage / "fonts" / name)
            print(f"staged Latin Modern from {lm_dir}")

        print(f"using {browser}")
        print(f"staged {len(set(used))} figures in {stage}")
        print(f"spliced {n_spliced - len(missing)} figures into the document")
        print(f"tables: {kept} kept whole, {splittable} long enough to split")
        if missing:
            print(f"  WARNING: {len(missing)} referenced figure(s) not found:")
            for m in missing:
                print(f"    {m}")

        staged_pdf = stage / "manuscript.pdf"
        print_to_pdf(browser, stage / "manuscript.html", staged_pdf,
                     stage / "profile", args.timeout)

        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_pdf, out)
        print(f"wrote {out}")
    finally:
        if not args.keep_stage:
            shutil.rmtree(stage, ignore_errors=True)

    if report:
        # A bare centred number, as an article has; no running title.
        print(f"stamped {stamp_plain_page_numbers(out)} pages in {out}")
    else:
        # Page numbers and the running footer, as the other reports get them.
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "_stamp_pages.py"),
             str(out), args.title],
            check=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
