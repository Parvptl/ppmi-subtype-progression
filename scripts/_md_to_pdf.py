"""Render the explanatory report from Markdown to a printable HTML file.

Written for this project rather than pulled in as a dependency: `pandoc` is not
installed here and `pyproject.toml` deliberately keeps the dependency list
small. This handles exactly the Markdown the report uses -- headings, tables,
fenced code, blockquotes, lists, bold/italic/code spans and rules -- and nothing
more.

    python scripts/_md_to_pdf.py INPUT.md OUTPUT.html
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.2pt; line-height: 1.5; color: #1a1a1a;
  margin: 0; padding: 0; -webkit-print-color-adjust: exact;
}
h1 {
  font-size: 19pt; margin: 0 0 10pt; padding-bottom: 6pt;
  border-bottom: 2.5px solid #1f4e79; color: #1f4e79;
  page-break-before: always; page-break-after: avoid;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
  font-size: 14pt; margin: 20pt 0 7pt; color: #1f4e79;
  border-bottom: 1px solid #c8d6e5; padding-bottom: 3pt;
  page-break-after: avoid;
}
h3 { font-size: 11.6pt; margin: 14pt 0 5pt; color: #2c5f8d; page-break-after: avoid; }
h4 { font-size: 10.6pt; margin: 11pt 0 4pt; color: #37474f; page-break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 8pt; padding-left: 20pt; }
li { margin-bottom: 3pt; }
strong { color: #0d2d4d; }
code {
  font-family: "Consolas", "Courier New", monospace; font-size: 9pt;
  background: #f0f3f6; padding: 1px 4px; border-radius: 3px; color: #1f4e79;
}
pre {
  background: #f6f8fa; border: 1px solid #d8dee4; border-left: 3px solid #1f4e79;
  border-radius: 4px; padding: 8pt 10pt; margin: 0 0 9pt;
  font-family: "Consolas", "Courier New", monospace; font-size: 8.8pt;
  line-height: 1.42; overflow-x: hidden; white-space: pre-wrap;
  word-break: break-word; page-break-inside: avoid;
}
pre code { background: none; padding: 0; color: #24292f; }
blockquote {
  margin: 0 0 9pt; padding: 7pt 11pt; background: #eef4fa;
  border-left: 3.5px solid #1f4e79; color: #16324f; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 11pt; font-size: 8.9pt;
}
/* Long tables are allowed to split across pages -- forbidding it leaves large
   blank gaps. The header row repeats on each continuation page, and individual
   rows are kept whole. */
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th {
  background: #1f4e79; color: #fff; font-weight: 600; text-align: left;
  padding: 5pt 6pt; border: 1px solid #1f4e79; vertical-align: bottom;
}
th strong, th em { color: #fff; }
td { padding: 4.5pt 6pt; border: 1px solid #ccd6e0; vertical-align: top; }
tbody tr:nth-child(even) { background: #f4f7fa; }
td.num, th.num { text-align: right; white-space: nowrap; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 14pt 0; }
a { color: #1f4e79; text-decoration: none; }
.subtitle { color: #55606b; font-size: 10pt; margin-top: -4pt; }
"""

_INLINE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{html.escape(m.group(1))}</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
]


def inline(text: str) -> str:
    """Apply inline formatting. Code spans are escaped first so they stay literal."""
    placeholders: list[str] = []

    def stash(m):
        placeholders.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    for pattern, repl in _INLINE[1:]:
        text = pattern.sub(repl, text)
    text = text.replace("--", "&ndash;") if False else text
    return re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)


def is_numeric(cell: str) -> bool:
    bare = re.sub(r"[*`\s]", "", cell)
    return bool(re.fullmatch(r"[-+<>=~.,0-9eE%\[\]/]+", bare)) and any(c.isdigit() for c in bare)


def render_table(rows: list[str]) -> str:
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    aligns = [c.strip() for c in rows[1].strip().strip("|").split("|")]
    body = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows[2:]]

    right = [a.endswith(":") and not a.startswith(":") for a in aligns]
    for col in range(len(header)):
        if not right[col]:
            vals = [r[col] for r in body if col < len(r) and r[col]]
            right[col] = bool(vals) and all(is_numeric(v) for v in vals)

    out = ["<table><thead><tr>"]
    for i, cell in enumerate(header):
        out.append(f'<th class="num">{inline(cell)}</th>' if right[i]
                   else f"<th>{inline(cell)}</th>")
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        for i in range(len(header)):
            cell = row[i] if i < len(row) else ""
            out.append(f'<td class="num">{inline(cell)}</td>' if right[i]
                       else f"<td>{inline(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    list_stack: list[str] = []

    def close_lists():
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_lists()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and \
                re.fullmatch(r"\|[\s:\-|]+\|", lines[i + 1].strip()):
            close_lists()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(render_table(block))
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            close_lists()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            close_lists()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            text = " ".join(b for b in block if b)
            out.append(f"<blockquote><p>{inline(text)}</p></blockquote>")
            continue

        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if m:
            tag = "ul" if m.group(2) in ("-", "*", "+") else "ol"
            if not list_stack or list_stack[-1] != tag:
                close_lists()
                out.append(f"<{tag}>")
                list_stack.append(tag)
            # Lazy continuation: indented, non-blank lines that are not a new
            # item belong to this bullet. Without this the list closes mid-way
            # and an ordered list restarts its numbering at 1.
            item = [m.group(3)]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                if re.match(r"^\s*([-*+]|\d+\.)\s+", nxt) or re.match(
                        r"^\s*(#{1,6}\s|\||>|```|-{3,}$)", nxt):
                    break
                item.append(nxt.strip())
                i += 1
            out.append(f"<li>{inline(' '.join(item))}</li>")
            continue

        if not stripped:
            close_lists()
            i += 1
            continue

        close_lists()
        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(#{1,6}\s|[-*+]\s|\d+\.\s|\||>|```|-{3,}$)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")

    close_lists()
    return "\n".join(out)


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    body = convert(src.read_text(encoding="utf-8"))
    dst.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(src.stem)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>", encoding="utf-8")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
