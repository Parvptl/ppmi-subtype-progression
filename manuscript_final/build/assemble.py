#!/usr/bin/env python3
"""Assemble manuscript_final/full_research_paper.md from build/sections/*.md.

The section files are the editable source. Inside them, `{{table:N}}` is
replaced by table N of tables/TABLE_INDEX.csv rendered as a Markdown table
with its caption, and `{{figure:N}}` by figure N of figures/FIGURE_INDEX.csv
as an image reference with its caption. The assembled file is what the PDF
builder and the number verifier read. Nothing outside manuscript_final/ is
written.
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MF = HERE.parent
SEC = HERE / "sections"
OUT = MF / "full_research_paper.md"


def load_index(path: Path, key: str) -> dict:
    with path.open() as fh:
        return {int(r[key]): r for r in csv.DictReader(fh)}


def md_table(csv_path: Path) -> str:
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "*(empty table)*"
    def esc(c: str) -> str:
        return c.replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(esc(c) for c in rows[0]) + " |"
    sep = "|" + "|".join("---" for _ in rows[0]) + "|"
    body = ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join([head, sep] + body)


def main() -> int:
    tables = load_index(MF / "tables" / "TABLE_INDEX.csv", "table")
    figures = load_index(MF / "figures" / "FIGURE_INDEX.csv", "figure")
    used_t, used_f = set(), set()
    parts = []
    for f in sorted(SEC.glob("*.md")):
        parts.append(f.read_text().rstrip() + "\n")
    text = "\n".join(parts)

    def rep_table(m):
        n = int(m.group(1))
        if n not in tables:
            raise SystemExit(f"unknown table {n}")
        used_t.add(n)
        t = tables[n]
        src = t["sources"]
        out = [f"**Table {n}.** {t['caption']}", "", md_table(MF / "tables" / t["file"]), "",
               f"*Machine-readable copy: `tables/{t['file']}`. Source: `{src}`.*"]
        return "\n".join(out)

    def rep_fig(m):
        n = int(m.group(1))
        if n not in figures:
            raise SystemExit(f"unknown figure {n}")
        used_f.add(n)
        f = figures[n]
        return "\n".join([f"![Figure {n}](figures/{f['file']})", "",
                          f"**Figure {n}.** {f['caption']} *(Source: `{f['source']}`.)*"])

    text = re.sub(r"\{\{table:(\d+)\}\}", rep_table, text)
    text = re.sub(r"\{\{figure:(\d+)\}\}", rep_fig, text)
    OUT.write_text(text)
    words = len(re.findall(r"\b\w+\b", text))
    print(f"wrote {OUT.name}: {len(text.splitlines())} lines, about {words:,} words")
    unused_t = sorted(set(tables) - used_t); unused_f = sorted(set(figures) - used_f)
    if unused_t:
        print("tables not referenced:", unused_t)
    if unused_f:
        print("figures not referenced:", unused_f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
