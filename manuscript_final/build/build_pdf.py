#!/usr/bin/env python3
"""Render manuscript_final/full_research_paper.md to full_research_paper.pdf.

A Markdown-to-PDF renderer in reportlab, adapted from the repository's
scripts/86_build_manuscript_pdf.py so that the manuscript stays within the
existing Markdown-based workflow (no pandoc or LaTeX is required). It handles
headings, paragraphs, bullet and numbered lists, block quotes, fenced and
indented code, pipe tables (split across pages, header repeated), images and
inline bold / italic / code. Nothing here computes a result.
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, ListFlowable, ListItem)

HERE = Path(__file__).resolve().parent
MF = HERE.parent
SRC = MF / "full_research_paper.md"
OUT = MF / "full_research_paper.pdf"
INK = colors.HexColor("#1a1a1a"); ACCENT = colors.HexColor("#1f4e79"); GREY = colors.HexColor("#5a6066")


def register_fonts() -> str:
    for d in [Path("/usr/share/fonts/dejavu"), Path("/usr/share/fonts/truetype/dejavu")]:
        try:
            pdfmetrics.registerFont(TTFont("DJ", d / "DejaVuSans.ttf"))
            pdfmetrics.registerFont(TTFont("DJ-B", d / "DejaVuSans-Bold.ttf"))
            pdfmetrics.registerFont(TTFont("DJ-I", d / "DejaVuSans-Oblique.ttf"))
            pdfmetrics.registerFont(TTFont("DJ-BI", d / "DejaVuSans-BoldOblique.ttf"))
            pdfmetrics.registerFont(TTFont("DJ-M", d / "DejaVuSansMono.ttf"))
            pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I", boldItalic="DJ-BI")
            return "DJ"
        except Exception:
            continue
    print("DejaVu fonts not found; falling back to Helvetica", file=sys.stderr)
    return "Helvetica"


FONT = register_fonts()
MONO = "DJ-M" if FONT == "DJ" else "Courier"


def inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+?)`", lambda m: f'<font face="{MONO}" size="7.6">{m.group(1)}</font>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\*\w])\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = s.replace("\u2070", "<super>0</super>")
    return s


def mk(name, **kw):
    kw.setdefault("fontName", FONT); kw.setdefault("textColor", INK)
    return ParagraphStyle(name=name, **kw)


ST = {
    "title": mk("title", fontName=FONT + "-B" if FONT == "DJ" else "Helvetica-Bold", fontSize=16, leading=21, alignment=TA_CENTER, spaceAfter=10),
    "authors": mk("authors", fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=3),
    "affil": mk("affil", fontSize=9.4, leading=13, alignment=TA_CENTER, textColor=GREY, spaceAfter=14),
    "h1": mk("h1", fontName=FONT + "-B" if FONT == "DJ" else "Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=16, spaceAfter=7, textColor=ACCENT),
    "h2": mk("h2", fontName=FONT + "-B" if FONT == "DJ" else "Helvetica-Bold", fontSize=11, leading=14.5, spaceBefore=12, spaceAfter=5),
    "h3": mk("h3", fontName=FONT + "-BI" if FONT == "DJ" else "Helvetica-BoldOblique", fontSize=10, leading=13.5, spaceBefore=9, spaceAfter=4),
    "h4": mk("h4", fontName=FONT + "-B" if FONT == "DJ" else "Helvetica-Bold", fontSize=9.4, leading=12.5, spaceBefore=7, spaceAfter=3),
    "body": mk("body", fontSize=9.4, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6),
    "quote": mk("quote", fontSize=8.9, leading=12.6, leftIndent=10, rightIndent=4, spaceAfter=7, alignment=TA_JUSTIFY, textColor=GREY),
    "mono": mk("mono", fontName=MONO, fontSize=7.6, leading=10.2, leftIndent=8, spaceAfter=7),
    "caption": mk("caption", fontSize=8.6, leading=12, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=8),
    "cell": mk("cell", fontSize=6.6, leading=8.4, alignment=TA_LEFT),
    "cellh": mk("cellh", fontName=FONT + "-B" if FONT == "DJ" else "Helvetica-Bold", fontSize=6.6, leading=8.4),
    "li": mk("li", fontSize=9.4, leading=13.4, alignment=TA_LEFT, spaceAfter=2),
}


def render_table(rows: list[list[str]], avail: float):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    fs = 6.6 if ncol <= 8 else (5.9 if ncol <= 11 else 5.2)
    cell = ParagraphStyle("c2", parent=ST["cell"], fontSize=fs, leading=fs * 1.28)
    cellh = ParagraphStyle("ch2", parent=ST["cellh"], fontSize=fs, leading=fs * 1.28)
    data = [[Paragraph(inline(c), cellh) for c in rows[0]]] + [[Paragraph(inline(c), cell) for c in r] for r in rows[1:]]
    def longest(j):
        best = 0.0
        for r in rows:
            txt = re.sub(r"[*`]", "", r[j])
            for word in re.split(r"[\s/]+", txt):
                best = max(best, pdfmetrics.stringWidth(word, FONT, fs))
            best = max(best, pdfmetrics.stringWidth(txt, FONT, fs) * 0.35)
        return best + 7.0
    want = [longest(j) for j in range(ncol)]
    total = sum(want)
    widths = [w * avail / total for w in want] if total > avail else [w + (avail - total) * w / total for w in want]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT), ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7f9")]),
    ]))
    return t


def main() -> int:
    lines = SRC.read_text().splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Manuscript")
    authors = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Authors:")), "")
    affil = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Affiliation:")), "")

    def decorate(canvas, doc):
        canvas.saveState(); canvas.setFont(FONT, 8); canvas.setFillColor(colors.HexColor("#8a9099"))
        canvas.drawCentredString(A4[0] / 2.0, 11 * mm, str(doc.page))
        if doc.page > 1:
            canvas.drawString(20 * mm, A4[1] - 12 * mm, title[:110])
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=19 * mm, bottomMargin=19 * mm,
                            title=title, author=authors)
    avail = doc.width; maxh = doc.height - 30 * mm
    flow = [Spacer(1, 22 * mm), Paragraph(inline(title), ST["title"]), Spacer(1, 4),
            Paragraph(inline(authors), ST["authors"]), Paragraph(inline(affil), ST["affil"])]
    i, buf = 0, []
    list_items, list_kind = [], None

    def flush_par():
        nonlocal buf
        if buf:
            txt = " ".join(x.strip() for x in buf).strip()
            if txt:
                flow.append(Paragraph(inline(txt), ST["body"]))
            buf = []

    def flush_list():
        nonlocal list_items, list_kind
        if list_items:
            items = [ListItem(Paragraph(inline(t), ST["li"]), leftIndent=12) for t in list_items]
            flow.append(ListFlowable(items, bulletType="bullet" if list_kind == "ul" else "1", start=None if list_kind == "ul" else 1,
                                     bulletFontName=FONT, bulletFontSize=8, leftIndent=14))
            flow.append(Spacer(1, 4))
            list_items, list_kind = [], None

    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        if s.startswith("# ") and not flow[-1:] == []:
            flush_par(); flush_list(); i += 1; continue           # title already placed
        if s.startswith("Authors:") or s.startswith("Affiliation:"):
            flush_par(); flush_list(); i += 1; continue
        if s.startswith("```"):
            flush_par(); flush_list(); code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            flow.append(Paragraph("<br/>".join(inline(c).replace(" ", "&nbsp;") for c in code) or "&nbsp;", ST["mono"]))
            i += 1; continue
        m = re.match(r"^(#{2,4})\s+(.*)$", s)
        if m:
            flush_par(); flush_list()
            level = len(m.group(1)); style = {2: "h1", 3: "h2", 4: "h3"}[level]
            if level == 2 and m.group(2).strip().lower() in ("introduction", "methods", "results", "discussion", "references", "supplementary material"):
                flow.append(PageBreak())
            flow.append(Paragraph(inline(m.group(2)), ST[style])); i += 1; continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            flush_par(); flush_list()
            fp = MF / m.group(2)
            if fp.exists():
                iw, ih = ImageReader(str(fp)).getSize()
                w = min(avail, 160 * mm); h = w * ih / iw
                if h > maxh:
                    h = maxh; w = h * iw / ih
                flow.append(Image(str(fp), width=w, height=h)); flow.append(Spacer(1, 3))
            else:
                flow.append(Paragraph(f"<i>missing figure {m.group(2)}</i>", ST["body"]))
            i += 1; continue
        if s.startswith("|"):
            flush_par(); flush_list(); tb = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1; tb.append(lines[i])
            rows = [[c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in tb if not re.match(r"^\|[\s:\-|]+\|$", r.strip())]
            if rows:
                flow.append(render_table(rows, avail)); flow.append(Spacer(1, 6))
            i += 1; continue
        if s.startswith("> "):
            flush_par(); flush_list(); q = [s[2:]]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith(">"):
                i += 1; q.append(lines[i].strip().lstrip(">").strip())
            flow.append(Paragraph(inline(" ".join(q)), ST["quote"])); i += 1; continue
        m = re.match(r"^(\d+)\.\s+(.*)$", s); mb = re.match(r"^[-*]\s+(.*)$", s)
        if m or mb:
            flush_par()
            kind = "ol" if m else "ul"
            if list_kind and list_kind != kind:
                flush_list()
            list_kind = kind
            text = (m.group(2) if m else mb.group(1))
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip() and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i + 1]):
                i += 1; text += " " + lines[i].strip()
            list_items.append(text); i += 1; continue
        if ln.startswith("    ") and s:
            flush_par(); flush_list(); code = [ln[4:]]
            while i + 1 < len(lines) and lines[i + 1].startswith("    "):
                i += 1; code.append(lines[i][4:])
            flow.append(Paragraph("<br/>".join(inline(c).replace(" ", "&nbsp;") for c in code), ST["mono"])); i += 1; continue
        if s == "---":
            flush_par(); flush_list(); i += 1; continue
        if not s:
            flush_par(); flush_list(); i += 1; continue
        if s.startswith("**Table ") or s.startswith("**Figure "):
            flush_par(); flush_list(); flow.append(Paragraph(inline(s), ST["caption"])); i += 1; continue
        if s.startswith("*Machine-readable copy") or s.startswith("*Source"):
            flush_par(); flow.append(Paragraph(inline(s), ST["caption"])); i += 1; continue
        buf.append(s); i += 1
    flush_par(); flush_list()
    doc.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
