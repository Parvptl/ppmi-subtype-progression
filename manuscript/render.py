"""Render MANUSCRIPT.md to a self-contained HTML document, figures included.

The project already has a Markdown renderer, ``scripts/_md_to_pdf.py``, used for
the progress reports. It handles headings, tables, lists, code and rules but not
images, because none of the earlier reports embedded any. Rather than change a
script other reports depend on, this wrapper reuses its ``convert`` and ``CSS``
and adds exactly one thing: Markdown image syntax becomes a ``<figure>`` with
the PNG inlined as a base64 data URI, so the result is a single file that can be
e-mailed or opened anywhere with no accompanying image directory.

PDF: the repository's historical route was this HTML printed by headless Chrome
and then page-stamped with ``scripts/_stamp_pages.py``. Chrome is not installed
in this environment, so ``--pdf`` instead uses PyMuPDF, which is already a
dependency here. If that is unavailable the HTML is still written and the
command says so rather than failing.

    python manuscript/render.py [--pdf]
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_renderer():
    """Import the project's Markdown renderer by path (its name starts with _)."""
    path = ROOT / "scripts" / "_md_to_pdf.py"
    spec = importlib.util.spec_from_file_location("_md_to_pdf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_md_to_pdf"] = mod
    spec.loader.exec_module(mod)
    return mod


FIGURE_CSS = """
figure { margin: 14pt 0 16pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #e3e2df; }
figcaption {
  font-size: 8.8pt; color: #52514e; text-align: left;
  margin-top: 5pt; line-height: 1.45;
}
/* The figure number leads every caption in bold; give it the heading blue.
   <strong> is what the project's inline formatter emits for **...**. */
figcaption b, figcaption strong { color: #1f4e79; }
.missing-figure {
  border: 1px dashed #c0392b; color: #c0392b; padding: 8pt;
  font-size: 9pt; text-align: left;
}
"""

IMAGE = re.compile(r"^!\[(?P<caption>.*?)\]\((?P<path>[^)]+)\)\s*$")

#: Stand-in left in the Markdown where a figure belongs, swapped for real HTML
#: after conversion by :func:`splice_figures`.
#:
#: Raw ``<figure>`` HTML cannot be put into the Markdown and left for the
#: renderer to pass through: ``_md_to_pdf.convert`` has no raw-HTML branch, so
#: an unrecognised line falls through to its paragraph handler, which runs
#: ``inline()`` -> ``html.escape()`` and turns the tags into visible text. The
#: placeholder is deliberately bare alphanumerics so that escaping leaves it
#: untouched and it can be found again afterwards.
MARK = "FIGUREPLACEHOLDER{}ENDFIGUREPLACEHOLDER"
MARK_RE = re.compile(
    r"<p>\s*FIGUREPLACEHOLDER(\d+)ENDFIGUREPLACEHOLDER\s*</p>")


def embed_images(md: str, base: Path,
                 inline_data: bool = True) -> tuple[str, list[str], list[str]]:
    """Swap image lines for placeholders. Returns (md, blocks, missing).

    ``blocks`` holds the HTML for each placeholder in order, to be spliced in
    by :func:`splice_figures` once the Markdown has been converted.

    With ``inline_data`` the PNG is carried as a base64 data URI, so the HTML
    stands alone with no image directory. Without it the figure keeps its
    relative path, which is what a browser's print-to-PDF path wants: the
    base64 form makes a multi-megabyte document that headless Chrome will not
    finish laying out.

    A referenced figure that does not exist is rendered as a visible red box
    rather than silently dropped, so a stale reference cannot pass unnoticed
    into a document someone is about to present.
    """
    # Captions carry Markdown emphasis -- every one opens with **Figure n.** --
    # so they go through the project's inline formatter rather than being
    # escaped wholesale, which would print the asterisks literally.
    inline = load_renderer().inline

    out, blocks, missing = [], [], []
    for line in md.splitlines():
        m = IMAGE.match(line.strip())
        if not m:
            out.append(line)
            continue
        rel = m.group("path")
        path = (base / rel).resolve()
        caption = m.group("caption")
        if not path.exists():
            missing.append(rel)
            blocks.append(f'<div class="missing-figure">MISSING FIGURE: '
                          f'{html_mod.escape(rel)}</div>')
        else:
            if inline_data:
                uri = base64.b64encode(path.read_bytes()).decode("ascii")
                src = f"data:image/png;base64,{uri}"
            else:
                src = html_mod.escape(rel)
            # alt stays plain text and escaped; the caption is formatted.
            alt = html_mod.escape(re.sub(r"[*`]", "", caption), quote=True)
            blocks.append(
                f'<figure><img src="{src}" alt="{alt}"/>'
                f'<figcaption>{inline(caption)}</figcaption></figure>')
        out.append(MARK.format(len(blocks) - 1))
    return "\n".join(out), blocks, missing


def splice_figures(html_text: str, blocks: list[str]) -> tuple[str, int]:
    """Replace each placeholder paragraph with its figure. Returns (html, n).

    Every placeholder must be found: one left behind would print as a nonsense
    word where a figure should be, so an unmatched placeholder is an error
    rather than something to tolerate.
    """
    seen: list[int] = []

    def sub(m: re.Match) -> str:
        idx = int(m.group(1))
        seen.append(idx)
        return blocks[idx]

    out = MARK_RE.sub(sub, html_text)
    if len(seen) != len(blocks) or "FIGUREPLACEHOLDER" in out:
        raise RuntimeError(
            f"spliced {len(seen)} of {len(blocks)} figures; the renderer's "
            "output did not have the expected placeholder paragraphs")
    return out, len(seen)


#: A rendered manuscript of this length is a few dozen pages. Anything far
#: beyond that means the layout engine has failed rather than succeeded.
MAX_SENSIBLE_PAGES = 80


def to_pdf(html_text: str, path: Path, n_expected_figures: int) -> bool:
    """Attempt a PDF with PyMuPDF, and reject the result if it is not usable.

    PyMuPDF's Story engine does not resolve ``data:`` image URIs and paginates
    this document very poorly, so it produces a long, figureless file. That is
    worse than no PDF, because it looks like a deliverable. The output is
    therefore checked for page count and embedded images and deleted unless it
    passes; the caller then reports that no PDF was produced.
    """
    try:
        import pymupdf
    except ImportError:
        return False

    tmp = path.with_suffix(".tmp.pdf")
    try:
        story = pymupdf.Story(html=html_text)
        writer = pymupdf.DocumentWriter(str(tmp))
        rect = pymupdf.paper_rect("a4")
        area = rect + (50, 50, -50, -50)
        more = True
        while more:
            device = writer.begin_page(rect)
            more, _ = story.place(area)
            story.draw(device)
            writer.end_page()
        writer.close()

        with pymupdf.open(tmp) as doc:
            pages = doc.page_count
            images = sum(len(p.get_images()) for p in doc)
        if pages > MAX_SENSIBLE_PAGES or images < n_expected_figures:
            print(f"  rejected the generated PDF: {pages} pages, "
                  f"{images} embedded images "
                  f"(expected <= {MAX_SENSIBLE_PAGES} pages and "
                  f">= {n_expected_figures} images)")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
        return True
    except Exception as exc:                        # noqa: BLE001
        print(f"  PDF generation failed: {type(exc).__name__}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(HERE / "MANUSCRIPT.md"))
    ap.add_argument("--html", default=str(HERE / "MANUSCRIPT.html"))
    ap.add_argument("--pdf", action="store_true",
                    help="also attempt a PDF (layout is cruder than the HTML)")
    args = ap.parse_args()

    src = Path(args.source)
    md = src.read_text(encoding="utf-8")
    md, blocks, missing = embed_images(md, src.parent)

    renderer = load_renderer()
    body, n_figs = splice_figures(renderer.convert(md), blocks)
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html_mod.escape(src.stem)}</title>"
        f"<style>{renderer.CSS}{FIGURE_CSS}</style></head>"
        f"<body>{body}</body></html>")

    out_html = Path(args.html)
    out_html.write_text(document, encoding="utf-8")
    # n_figs counts every spliced block, including any red missing-figure box.
    print(f"wrote {out_html.relative_to(ROOT)}  "
          f"({n_figs - len(missing)} figures embedded)")
    if missing:
        print(f"  WARNING: {len(missing)} referenced figure(s) not found:")
        for m in missing:
            print(f"    {m}")

    if args.pdf:
        out_pdf = out_html.with_suffix(".pdf")
        if to_pdf(document, out_pdf, n_figs):
            print(f"wrote {out_pdf.relative_to(ROOT)}")
        else:
            print("  no PDF written. The repository's usual route is to open "
                  "the HTML and print to PDF (that is how the existing reports "
                  "in report/ were produced); headless Chrome, weasyprint and "
                  "pdflatex are all absent from this environment.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
