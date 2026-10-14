"""Stamp page numbers and a running footer onto a rendered PDF.

Chrome's headless `--print-to-pdf` does not reliably emit its header/footer
template, so the footer is added afterwards with PyMuPDF. The bottom margin in
the print CSS is sized to leave room for this.

    python scripts/_stamp_pages.py FILE.pdf "Short title"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf


def main() -> int:
    path = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else path.stem

    doc = pymupdf.open(path)
    total = doc.page_count
    grey = (0.42, 0.45, 0.50)
    rule = (0.80, 0.84, 0.88)

    for n, page in enumerate(doc, start=1):
        width, height = page.rect.width, page.rect.height
        y = height - 30
        page.draw_line(pymupdf.Point(46, y - 9), pymupdf.Point(width - 46, y - 9),
                       color=rule, width=0.6)
        page.insert_text(pymupdf.Point(46, y + 2), title,
                         fontname="helv", fontsize=7.5, color=grey)
        label = f"Page {n} of {total}"
        page.insert_text(
            pymupdf.Point(width - 46 - pymupdf.get_text_length(
                label, fontname="helv", fontsize=7.5), y + 2),
            label, fontname="helv", fontsize=7.5, color=grey)

    tmp = path.with_suffix(".stamped.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(path)
    print(f"stamped {total} pages in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
