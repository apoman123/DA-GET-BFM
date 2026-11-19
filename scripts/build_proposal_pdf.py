"""Render the research proposal (Markdown, Traditional Chinese) to a readable A4 PDF.

Usage::

    python scripts/build_proposal_pdf.py docs/proposal/DA-GET-BFM_proposal_v4.md \
        docs/proposal/implementation_status.md -o docs/proposal/DA-GET-BFM_proposal_v4.pdf

The Markdown files are concatenated in order, converted to HTML with
Python-Markdown (tables, fenced code, table of contents), and typeset with
WeasyPrint. Neither package is a project dependency: install them where you run
this script (``pip install markdown weasyprint``). CJK text needs a Noto Sans
CJK TC (or similar) font installed on the system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STYLE = """
@page {
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8.5pt; color: #666; }
  @top-right { content: "DA-GET-BFM Proposal v4"; font-size: 8pt; color: #999; }
}
@page :first { @top-right { content: none; } }
html { font-family: "Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans", sans-serif; font-size: 10pt; line-height: 1.6; color: #1d1d1f; }
h1 { font-size: 21pt; margin: 0 0 4pt 0; color: #0b3d63; line-height: 1.25; }
h2 { font-size: 14.5pt; color: #0b3d63; border-bottom: 1.5pt solid #0b3d63; padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 12pt; color: #14507f; margin-top: 13pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 10pt; page-break-after: avoid; }
p { margin: 4pt 0 6pt 0; text-align: justify; }
strong { color: #111; }
code { font-family: "Noto Sans Mono CJK TC", "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: #f2f4f7; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f2f4f7; padding: 6pt 8pt; border-radius: 3pt; font-size: 8.4pt; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 8.8pt; page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th { background: #0b3d63; color: white; font-weight: 600; text-align: left; padding: 4pt 5pt; }
td { border-bottom: 0.6pt solid #d0d7de; padding: 3.5pt 5pt; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fb; }
ul, ol { margin: 3pt 0 6pt 0; padding-left: 16pt; }
li { margin: 1.5pt 0; }
hr { border: none; border-top: 0.8pt solid #c8d1db; margin: 10pt 0; }
blockquote { border-left: 3pt solid #9cb8d3; margin: 6pt 0; padding: 2pt 8pt; color: #333; background: #f5f8fb; }
.toc { border: 0.8pt solid #c8d1db; background: #f7f9fb; padding: 6pt 12pt; margin: 8pt 0 12pt 0; font-size: 9pt; }
.toc ul { padding-left: 12pt; }
.toc > ul > li > ul { display: none; }
.toc a { color: #14507f; text-decoration: none; }
a { color: #14507f; text-decoration: none; }
.page-break { page-break-before: always; }
"""


def render(markdown_files: list[Path], output: Path) -> None:
    """Concatenate Markdown files and write the PDF.

    Args:
        markdown_files (list[Path]): Proposal sections in order.
        output (Path): PDF path.

    Returns:
        None: Writes ``output``.
    """
    import markdown
    from weasyprint import HTML

    parts = []
    for index, path in enumerate(markdown_files):
        text = path.read_text(encoding="utf-8")
        if index:
            text = '<div class="page-break"></div>\n\n' + text
        parts.append(text)
    source = "\n\n".join(parts)
    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3", "title": "目錄"}},
    )
    # Put a table of contents after the title block (the first horizontal rule).
    toc = markdown.Markdown(extensions=["toc"], extension_configs={"toc": {"toc_depth": "2-2"}})
    toc.convert(source)
    if "<hr />" in body:
        body = body.replace("<hr />", f'<div class="toc"><strong>目錄</strong>{toc.toc}</div>', 1)
    html = f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>DA-GET-BFM Proposal v4</title><style>{STYLE}</style></head><body>{body}</body></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(markdown_files[0].parent)).write_pdf(output)


def main() -> None:
    """Parse arguments and render the PDF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("markdown", nargs="+", type=Path, help="Markdown files, concatenated in order")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output PDF path")
    arguments = parser.parse_args()
    render(arguments.markdown, arguments.output)


if __name__ == "__main__":
    main()
