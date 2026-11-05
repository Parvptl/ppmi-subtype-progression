#!/usr/bin/env python3
"""Render the assembled manuscript to a review-ready PDF.

Reads `manuscript/npj/MANUSCRIPT_FULL_20261104.md` and lays it out with
reportlab: title page, the five scientific sections, back matter, references,
then each figure with its legend and each table with its caption.

Internal working content is omitted from the PDF and only from the PDF: the
assembly note and the Asset status table are useful in the repository and
wrong in a manuscript. Author queries are kept and boxed, because an
unanswered declaration must be visible to the reader, not hidden.

Nothing here computes a result. Every number comes from the assembled
manuscript or from the formatted table CSVs that `scripts/80` writes.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parent.parent
NPJ = ROOT / "manuscript" / "npj"
SRC = NPJ / "MANUSCRIPT_FULL_20261104.md"
OUT = NPJ / "MANUSCRIPT_REVIEW_DRAFT.pdf"

FIGURES = {
    "1": ["fig1_participant_flow.png"],
    "2": ["fig2_label_instability.png"],
    "3": ["fig3_harmonised_comparison.png"],
    "4": ["fig4_margin_decomposition.png"],
    "5": ["fig5a_representation_dimensionality.png",
          "fig5b_outcome_discrimination.png"],
}
TABLES = {
    "1": "table1_cohort_characteristics.csv",
    "2": "table2_transition_probabilities.csv",
    "3": "table3_harmonised_comparison.csv",
    "4": "table4_two_class.csv",
    "5": "table5_composition.csv",
}
SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
       "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9"}
INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
WARM = colors.HexColor("#c2552c")
RULE = colors.HexColor("#c9ccd1")


def register_fonts() -> str:
    d = Path("/usr/share/fonts/dejavu")
    try:
        pdfmetrics.registerFont(TTFont("DJ", d / "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DJ-B", d / "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("DJ-I", d / "DejaVuSans-Oblique.ttf"))
        pdfmetrics.registerFont(TTFont("DJ-BI", d / "DejaVuSans-BoldOblique.ttf"))
        pdfmetrics.registerFont(TTFont("DJ-M", d / "DejaVuSansMono.ttf"))
        pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B",
                                      italic="DJ-I", boldItalic="DJ-BI")
        return "DJ"
    except Exception as exc:                                   # pragma: no cover
        print(f"  DejaVu unavailable ({exc}); falling back to Helvetica",
              file=sys.stderr)
        return "Helvetica"


def inline(s: str) -> str:
    """Markdown inline markup -> reportlab markup, superscripts included."""
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+?)`", r'<font face="DJ-M" size="8.4">\1</font>', s)
    # Runs of unicode superscript digits become real superscripts.
    def sup(m):
        return "<super>" + "".join(SUP[c] for c in m.group(0)) + "</super>"
    s = re.sub("[" + "".join(SUP) + "]+", sup, s)
    s = s.replace("\u02d2", "").replace("\u02d3", "")
    return s


def build_styles(font: str) -> dict:
    ss = getSampleStyleSheet()
    def mk(**kw):
        kw.setdefault("fontName", font)
        kw.setdefault("textColor", INK)
        return ParagraphStyle(**kw)
    return {
        "title": mk(name="t", parent=ss["Title"], fontName=font + "-B",
                    fontSize=17, leading=22, alignment=TA_CENTER,
                    spaceAfter=10, textColor=INK),
        "authors": mk(name="a", fontSize=11, leading=16, alignment=TA_CENTER,
                      spaceAfter=3),
        "affil": mk(name="af", fontSize=9.4, leading=13, alignment=TA_CENTER,
                    textColor=colors.HexColor("#555a60"), spaceAfter=3),
        "h1": mk(name="h1", fontName=font + "-B", fontSize=13.5, leading=17,
                 spaceBefore=16, spaceAfter=7, textColor=ACCENT),
        "h2": mk(name="h2", fontName=font + "-B", fontSize=11, leading=14.5,
                 spaceBefore=12, spaceAfter=5),
        "h3": mk(name="h3", fontName=font + "-BI", fontSize=10, leading=13.5,
                 spaceBefore=9, spaceAfter=4),
        "body": mk(name="b", fontSize=9.6, leading=14.2, alignment=TA_JUSTIFY,
                   spaceAfter=7),
        "abstract": mk(name="ab", fontSize=9.6, leading=14.4,
                       alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8,
                       spaceAfter=8),
        "legend": mk(name="lg", fontSize=9.0, leading=13, alignment=TA_JUSTIFY,
                     spaceBefore=5, spaceAfter=13),
        "ref": mk(name="r", fontSize=8.8, leading=12.4, leftIndent=13,
                  firstLineIndent=-13, spaceAfter=5, alignment=TA_JUSTIFY),
        "note": mk(name="q", fontSize=9.0, leading=13.2, leftIndent=10,
                   rightIndent=4, spaceBefore=2, spaceAfter=8,
                   alignment=TA_JUSTIFY,
                   textColor=colors.HexColor("#5a6066")),
        "kw": mk(name="kw", fontSize=9.0, leading=13, leftIndent=8,
                 rightIndent=8, spaceAfter=10),
        "mono": mk(name="m", fontName=font + "-M", fontSize=8.0, leading=11,
                   leftIndent=10, spaceAfter=8),
        "cell": mk(name="c", fontSize=7.6, leading=10),
        "cellh": mk(name="ch", fontName=font + "-B", fontSize=7.6, leading=10),
    }


def render_table(path: Path, st: dict, avail: float) -> Table | Paragraph:
    with path.open() as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return Paragraph("<i>table source empty</i>", st["body"])
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), st["cellh"]) for c in head]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
    ncol = len(head)
    #  Width in proportion to the widest content each column actually holds,
    #  measured rather than guessed: an even split breaks long labels like
    #  "Indeterminate" mid-word while leaving numeric columns half empty.
    def longest(j: int) -> float:
        best = 0.0
        for r in rows:
            if j < len(r):
                for word in re.split(r"[\s/]+", re.sub(r"[*`]", "", r[j])):
                    best = max(best, pdfmetrics.stringWidth(word, "DJ-B", 7.6))
                best = max(best, pdfmetrics.stringWidth(
                    re.sub(r"[*`]", "", r[j]), "DJ", 7.6) * 0.40)
        return best + 9.0
    want = [longest(j) for j in range(ncol)]
    total = sum(want)
    widths = ([w * avail / total for w in want] if total > avail
              else [w + (avail - total) * w / total for w in want])
    tb = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tb.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f7f9")]),
    ]))
    return tb


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1
    font = register_fonts()
    st = build_styles(font)
    lines = SRC.read_text().splitlines()

    def decorate(canvas, docu):
        canvas.saveState()
        canvas.setFont(font, 8.2)
        canvas.setFillColor(colors.HexColor("#8a9099"))
        canvas.drawCentredString(A4[0] / 2.0, 11 * mm, str(docu.page))
        if docu.page > 1:
            canvas.drawString(21 * mm, A4[1] - 12 * mm,
                              "Towards Longitudinal Prediction of Parkinson's "
                              "Disease Motor Subtypes")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=19 * mm, bottomMargin=19 * mm,
        title="Towards Longitudinal Prediction of Parkinson's Disease "
              "Motor Subtypes",
        author="Patel, Tewatia, Vasudevan, Sahasranand")
    avail = doc.width
    flow: list = []

    # ---- title block, taken from the manuscript's own first lines
    title = lines[0].lstrip("# ").strip()
    flow += [Spacer(1, 26 * mm), Paragraph(inline(title), st["title"]),
             Spacer(1, 5)]
    flow += [Paragraph(
                 "Parv Patel<super>1</super>, &nbsp;Mayank Tewatia<super>1</super>, "
                 "&nbsp;Varsha Vasudevan<super>2</super>, &nbsp;and "
                 "K. R. Sahasranand<super>1</super>", st["authors"]),
             Spacer(1, 2),
             Paragraph("<super>1</super> Indian Institute of Technology "
                       "Palakkad, India &nbsp;&nbsp;\u00b7&nbsp;&nbsp; "
                       "<super>2</super> NeuroComp", st["affil"]),
             Spacer(1, 16)]

    i, section, skip = 1, None, False
    buf: list[str] = []
    pending_tab: str | None = None

    def flush():
        nonlocal buf, pending_tab
        if buf:
            txt = " ".join(buf).strip()
            #  The title block already carries the author and affiliation
            #  lines; the markdown keeps them so it stands alone.
            if txt.startswith("**Authors.**") or txt.startswith("\u00b9 Indian"):
                buf = []
                return
            if txt:
                sty = st["abstract"] if section == "Abstract" else st["body"]
                if section == "References" and re.match(r"^\d+\.\s", txt):
                    sty = st["ref"]
                if pending_tab:
                    #  Caption above its table, and the pair kept on one page
                    #  where it fits; Table 3 is too tall and is allowed to run.
                    tb = render_table(NPJ / TABLES[pending_tab], st, avail)
                    group = [Paragraph(inline(txt), st["legend"]), tb,
                             Spacer(1, 11)]
                    flow.append(group[0] if pending_tab == "3"
                                else KeepTogether(group))
                    if pending_tab == "3":
                        flow.extend(group[1:])
                    pending_tab = None
                else:
                    flow.append(Paragraph(inline(txt), sty))
            buf = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith("## "):
            flush()
            if section == "Abstract":
                flow.append(Paragraph(
                    "<b>Keywords</b>&nbsp; Parkinson's disease &nbsp;\u00b7&nbsp; "
                    "motor subtype &nbsp;\u00b7&nbsp; longitudinal prediction "
                    "&nbsp;\u00b7&nbsp; label reliability &nbsp;\u00b7&nbsp; "
                    "hidden Markov model &nbsp;\u00b7&nbsp; PPMI", st["kw"]))
            section = s[3:].strip()
            # Working content: keep in the repository, not in the manuscript.
            skip = section == "Asset status"
            if not skip:
                if section in ("Abstract", "Introduction", "Results",
                               "Discussion", "Methods", "References",
                               "Figure legends", "Table captions"):
                    #  Only where a break earns its blank space: Methods
                    #  begins the back half, and the display items start
                    #  clean. References flow on, or they orphan a page.
                    if section in ("Methods", "Figure legends"):
                        flow.append(PageBreak())
                flow.append(Paragraph(inline(section), st["h1"]))
            i += 1
            continue
        if skip:
            i += 1
            continue
        if s.startswith("#### "):
            flush(); flow.append(Paragraph(inline(s[5:]), st["h3"])); i += 1; continue
        if s.startswith("### "):
            flush(); flow.append(Paragraph(inline(s[4:]), st["h2"])); i += 1; continue
        if s.startswith("> "):
            flush()
            q = [s[2:]]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith(">"):
                i += 1
                q.append(lines[i].strip().lstrip(">").strip())
            body = " ".join(q).strip()
            #  Declaration notes belong in the paper, set quietly: a reader
            #  must see that a statement is outstanding without the page
            #  looking like a working document. The assembly note is internal.
            if "Assembled by" not in body:
                flow.append(Paragraph(inline(body), st["note"]))
            i += 1
            continue
        if s.startswith("|"):
            flush()
            tb = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                tb.append(lines[i])
            rows = [[c.strip() for c in r.strip().strip("|").split("|")]
                    for r in tb if not re.match(r"^\|[\s:\-|]+\|$", r.strip())]
            data = [[Paragraph(inline(c), st["cellh"] if k == 0 else st["cell"])
                     for c in r] for k, r in enumerate(rows)]
            w = avail / max(len(rows[0]), 1)
            t = Table(data, colWidths=[w] * len(rows[0]), hAlign="LEFT")
            t.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, ACCENT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
                ("LINEBELOW", (0, -1), (-1, -1), 0.9, ACCENT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow += [t, Spacer(1, 9)]
            i += 1
            continue
        if ln.startswith("    ") and s:
            flush()
            code = [ln[4:]]
            while i + 1 < len(lines) and (lines[i + 1].startswith("    ")
                                          or not lines[i + 1].strip()):
                if not lines[i + 1].strip() and not (
                        i + 2 < len(lines) and lines[i + 2].startswith("    ")):
                    break
                i += 1
                code.append(lines[i][4:])
            flow.append(Paragraph(
                "<br/>".join(inline(c).replace(" ", "&nbsp;") for c in code),
                st["mono"]))
            i += 1
            continue
        if s == "---":
            flush(); i += 1; continue
        if not s:
            flush(); i += 1; continue

        #  Each reference is its own numbered entry. The list carries no
        #  blank lines between entries, so without this they merge into a
        #  single unreadable block.
        if section == "References" and re.match(r"^\d+\. ", s):
            flush()

        # Figure legend / table caption: emit the asset, then the text.
        mf = re.match(r"^\*\*Fig\. (\d)", s)
        mt = re.match(r"^\*\*Table (\d)", s)
        if section == "Figure legends" and mf:
            flush()
            for fn in FIGURES.get(mf.group(1), []):
                fp = NPJ / fn
                if fp.exists():
                    from reportlab.lib.utils import ImageReader
                    iw, ih = ImageReader(str(fp)).getSize()
                    w = min(avail, 165 * mm)
                    flow.append(Image(str(fp), width=w, height=w * ih / iw))
                    flow.append(Spacer(1, 4))
        if section == "Table captions" and mt:
            flush()
            if mt.group(1) in TABLES and (NPJ / TABLES[mt.group(1)]).exists():
                pending_tab = mt.group(1)
        buf.append(s)
        i += 1

    flush()
    doc.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
