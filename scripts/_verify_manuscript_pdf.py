"""Check a rendered MANUSCRIPT.pdf against MANUSCRIPT.md and MANUSCRIPT.html.

Reports, and exits non-zero on, anything the PDF is missing: headings, figure
captions, embedded figure images, table rows or references. Written because a
PDF that silently drops its figures still opens and still looks like a
deliverable -- which is exactly what the earlier render produced.

    python scripts/_verify_manuscript_pdf.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"


def norm(s: str) -> str:
    """Flatten the typographic variation introduced by the renderer.

    The PDF carries en dashes and curly quotes where the Markdown has ASCII (or
    the reverse, via &ndash;), and wraps lines wherever the column ends, so
    comparison has to be whitespace- and punctuation-insensitive.
    """
    s = unicodedata.normalize("NFKC", s)
    for a, b in [("−", "-"), ("–", "-"), ("—", "-"),
                 ("’", "'"), ("‘", "'"), ("“", '"'),
                 ("”", '"'), (" ", " ")]:
        s = s.replace(a, b)
    # Emphasis markers are dropped on both sides. Bold survives into the PDF as
    # a font change rather than as characters, so a literal ** left in the text
    # would otherwise be invisible here -- and that is exactly the defect that
    # made the first render print "**Figure 1.**".
    s = re.sub(r"[*`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def plain(s: str) -> str:
    """Strip Markdown emphasis so prose compares against rendered text."""
    return norm(re.sub(r"[*`]", "", s))


def load_make_pdf():
    """Import manuscript/make_pdf.py for its section-truncation rule, so the
    verifier and the renderer cannot disagree about where a cut falls."""
    import importlib.util
    path = MS / "make_pdf.py"
    spec = importlib.util.spec_from_file_location("ms_make_pdf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ms_make_pdf"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(MS / "MANUSCRIPT.pdf"))
    ap.add_argument("--through-section", type=int, default=None, metavar="N",
                    help="the PDF is an excerpt ending after section N; "
                         "compare it against the same truncation of the source")
    ap.add_argument("--keep-section", type=int, action="append", default=[],
                    metavar="N", help="section N was added back after the cut")
    ap.add_argument("--html", default=str(MS / "MANUSCRIPT.html"),
                    help="pass 'none' to skip the HTML cross-check")
    ap.add_argument("--footer", choices=("running", "plain"), default="running",
                    help="'running': a title plus 'Page n of m'; "
                         "'plain': a bare centred page number (report style)")
    ap.add_argument("--no-strapline", action="store_true",
                    help="report style drops the bold line under the title; "
                         "do not expect it in the PDF")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    md = (MS / "MANUSCRIPT.md").read_text(encoding="utf-8")
    if args.through_section is not None:
        md, dropped = load_make_pdf().truncate_sections(
            md, args.through_section, tuple(args.keep_section))
        kept = ", ".join(str(n) for n in sorted(set(args.keep_section)))
        print(f"excerpt: sections 1-{args.through_section}"
              + (f" plus {kept}" if kept else "")
              + f"; {len(dropped)} section(s) left out: {', '.join(dropped)}")
    doc = pymupdf.open(pdf_path)
    text = norm("\n".join(p.get_text() for p in doc))
    problems = 0

    print(f"{pdf_path.name}: {doc.page_count} pages, "
          f"{round(doc[0].rect.width / 72 * 25.4, 1)} x "
          f"{round(doc[0].rect.height / 72 * 25.4, 1)} mm")

    # --- headings -----------------------------------------------------------
    heads = re.findall(r"^#{1,6}\s+(.*)$", md, re.M)
    missing = [h for h in heads if plain(h) not in text]
    print(f"headings:          {len(heads) - len(missing)}/{len(heads)}")
    for h in missing:
        print(f"   MISSING HEADING: {plain(h)[:90]}")
    problems += len(missing)

    # --- figure captions ----------------------------------------------------
    caps = re.findall(r"^!\[(.*?)\]\(([^)]+)\)", md, re.M)
    cap_missing = [c for c, _ in caps if plain(c)[:70] not in text]
    print(f"figure captions:   {len(caps) - len(cap_missing)}/{len(caps)}")
    for c in cap_missing:
        print(f"   MISSING CAPTION: {plain(c)[:90]}")
    problems += len(cap_missing)

    # --- figure images actually embedded ------------------------------------
    images = [z for p in doc for z in p.get_image_info(xrefs=True)]
    print(f"embedded images:   {len(images)}/{len(caps)}")
    if len(images) < len(caps):
        print("   FEWER IMAGES THAN FIGURE REFERENCES")
        problems += len(caps) - len(images)
    lowres = [z for z in images
              if z["width"] / max((z["bbox"][2] - z["bbox"][0]) / 72, 1e-6) < 150]
    if lowres:
        print(f"   {len(lowres)} image(s) placed below 150 dpi")
        problems += len(lowres)

    # --- table rows ---------------------------------------------------------
    rows = [l for l in md.split("\n")
            if l.strip().startswith("|")
            and not re.fullmatch(r"\|[\s:|-]+\|", l.strip())]
    row_missing = []
    for row in rows:
        cells = [plain(c) for c in row.strip().strip("|").split("|")]
        cells = [c for c in cells if len(c) > 2]
        if not cells:
            continue
        if not all(c in text for c in cells):
            row_missing.append(row)
    print(f"table rows:        {len(rows) - len(row_missing)}/{len(rows)}")
    for r in row_missing:
        print(f"   MISSING ROW: {plain(r)[:90]}")
    problems += len(row_missing)

    # --- references ---------------------------------------------------------
    # An excerpt may stop before the reference list; then there is nothing to
    # check, and its absence is the point rather than a fault.
    parts = md.split("## 10. References")
    if len(parts) < 2:
        print("references:        n/a (not in this excerpt)")
    else:
        refs = parts[1].split("\n## ")[0]
        entries = [l for l in refs.split("\n")
                   if re.match(r"^\s*(\d+\.|[-*])\s+\S", l)]
        ref_missing = []
        for e in entries:
            body = plain(re.sub(r"^\s*(\d+\.|[-*])\s+", "", e))
            body = re.sub(r"[\[\]]", "", body)
            if len(body) < 15:
                continue
            if body[:55] not in text:
                ref_missing.append(body)
        print(f"references:        "
              f"{len(entries) - len(ref_missing)}/{len(entries)}")
        for r in ref_missing:
            print(f"   MISSING REF: {r[:90]}")
        problems += len(ref_missing)

    # --- the HTML, as a second opinion on figure count ----------------------
    # Only meaningful for the full document: the HTML is never an excerpt, so
    # comparing its figure count against a truncated PDF would be apples to
    # oranges.
    html_path = Path(args.html) if args.html != "none" else Path("none")
    if args.through_section is not None:
        print("html <figure>:     skipped (PDF is an excerpt)")
    elif html_path.exists():
        h = html_path.read_text(encoding="utf-8")
        n_html = h.count("<figure>")
        escaped = h.count("&lt;figure&gt;")
        left = h.count("FIGUREPLACEHOLDER")
        print(f"html <figure>:     {n_html}/{len(caps)}"
              f"  (escaped: {escaped}, leftover placeholders: {left})")
        if n_html != len(caps) or escaped or left:
            print("   HTML FIGURE MISMATCH")
            problems += 1

    # --- the strapline the report style removes -----------------------------
    if args.no_strapline:
        strap = next((l for l in md.splitlines()
                      if re.fullmatch(r"\*\*.+\*\*", l.strip())
                      and "project" in l.lower()), None)
        if strap:
            gone = plain(strap)[:50] not in text
            print(f"strapline removed: {'yes' if gone else 'NO'}")
            if not gone:
                problems += 1

    # --- stray markup -------------------------------------------------------
    # norm() is emphasis-insensitive so that matching works, which means an
    # unprocessed ** would slip past every check above. Look for it directly,
    # along with placeholders and escaped tags.
    raw = "\n".join(p.get_text() for p in doc)
    for token, label in [("**", "literal bold markers"),
                         ("FIGUREPLACEHOLDER", "unspliced figure placeholder"),
                         ("&lt;", "escaped HTML tag"),
                         ("<figure>", "raw figure tag"),
                         ("base64,", "raw base64 payload"),
                         ("MISSING FIGURE", "missing-figure box")]:
        n = raw.count(token)
        if n:
            print(f"   STRAY MARKUP: {n} x {label} ({token!r})")
            problems += n

    # --- page footers -------------------------------------------------------
    if args.footer == "running":
        stamped = sum(1 for p in doc
                      if re.search(r"Page \d+ of \d+", p.get_text()))
    else:
        # A bare number carries no distinguishing text, so look for a block in
        # the bottom strip of the page whose content is exactly that page's
        # number -- which also catches a number stamped on the wrong page.
        stamped = 0
        for n, p in enumerate(doc, start=1):
            foot = p.rect.height - 70
            if any(b[4].strip() == str(n) and b[1] >= foot
                   for b in p.get_text("blocks")):
                stamped += 1
    print(f"page numbers:      {stamped}/{doc.page_count} ({args.footer})")
    if stamped != doc.page_count:
        problems += doc.page_count - stamped

    print()
    print("ALL CHECKS PASSED" if problems == 0
          else f"{problems} PROBLEM(S) FOUND")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
