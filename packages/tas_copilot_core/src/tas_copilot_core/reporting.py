from __future__ import annotations

import html as _html_module
import json
import re
from pathlib import Path
from typing import Callable, Any


MarkdownRenderer = Callable[[dict[str, Any]], str]


def write_report_bundle(
    output_dir: str | Path,
    report: dict[str, Any],
    *,
    markdown_renderer: MarkdownRenderer,
    json_filename: str = "report.json",
    markdown_filename: str = "report.md",
    html_filename: str | None = None,
) -> dict[str, str]:
    """Write a workflow report as paired JSON and Markdown artifacts.

    When ``html_filename`` is given, the Markdown is also rendered as a
    styled standalone HTML page for browser-facing surfaces (day indexes,
    morning reports).
    """

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / json_filename
    markdown_path = output_path / markdown_filename
    artifact_free = {key: value for key, value in report.items() if key != "artifacts"}
    json_path.write_text(json.dumps(artifact_free, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_text = markdown_renderer(artifact_free)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    artifacts = {"json": str(json_path), "markdown": str(markdown_path)}
    if html_filename:
        html_path = output_path / html_filename
        html_path.write_text(render_markdown_report_html(markdown_text), encoding="utf-8")
        artifacts["html"] = str(html_path)
    return artifacts


_REPORT_HTML_STYLE = """
html{color-scheme:light;background:#ffffff;}
body{font-family:system-ui,-apple-system,sans-serif;margin:2rem auto;max-width:70rem;padding:0 1rem;color:#1f2937;background:#ffffff;}
h1,h2,h3{color:#111827;} h1{border-bottom:2px solid #e5e7eb;padding-bottom:0.3rem;}
table{border-collapse:collapse;width:100%;margin:0.75rem 0;} td,th{border:1px solid #e5e7eb;padding:0.4rem 0.6rem;text-align:left;font-size:0.92rem;}
th{background:#f4f4f5;} tr:nth-child(even) td{background:#fafafa;}
code{background:#f4f4f5;padding:0.1rem 0.3rem;border-radius:3px;font-size:0.9em;}
pre{background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:0.75rem;overflow-x:auto;} pre code{background:none;padding:0;}
li{margin:0.15rem 0;} .review-banner{color:#b45309;font-size:0.95rem;}
""".strip()


def render_markdown_report_html(markdown_text: str, *, title: str | None = None) -> str:
    """Render workflow-report Markdown (headers, lists, tables, fenced code,
    inline code/bold) as a standalone styled HTML page."""

    def inline(text: str) -> str:
        escaped = _html_module.escape(text, quote=False)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
        return escaped

    lines = markdown_text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    table: list[list[str]] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []
    page_title = title

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal table
        if table:
            head, *body = table
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(cell)}</th>" for cell in head) + "</tr></thead><tbody>")
            for row in body:
                out.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
            out.append("</tbody></table>")
            table = []

    for line in lines:
        stripped = line.strip()
        if in_code:
            if stripped.startswith("```"):
                out.append("<pre><code>" + _html_module.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                code_lines.append(line)
            continue
        if stripped.startswith("```"):
            flush_paragraph(); flush_list(); flush_table()
            in_code = True
            continue
        if stripped.startswith("|"):
            flush_paragraph(); flush_list()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            table.append(cells)
            continue
        flush_table()
        image = re.match(r"^!\[(.*)\]\(([^)]+)\)$", stripped)
        if image:
            flush_paragraph(); flush_list()
            alt = _html_module.escape(image.group(1), quote=True)
            src = _html_module.escape(image.group(2), quote=True)
            out.append(f"<figure><img src=\"{src}\" alt=\"{alt}\" style=\"max-width:100%;height:auto;\"><figcaption>{alt}</figcaption></figure>")
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush_paragraph(); flush_list()
            level = len(heading.group(1))
            text = heading.group(2)
            if page_title is None and level == 1:
                page_title = text
            out.append(f"<h{level}>{inline(text)}</h{level}>")
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
            continue
        if not stripped:
            flush_paragraph(); flush_list()
            continue
        paragraph.append(stripped)
    flush_paragraph(); flush_list(); flush_table()
    if in_code and code_lines:
        out.append("<pre><code>" + _html_module.escape("\n".join(code_lines)) + "</code></pre>")
    body = "\n".join(out)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>{_html_module.escape(page_title or 'TASCopilot report')}</title>"
        f"<style>{_REPORT_HTML_STYLE}</style></head><body>\n"
        f"{body}\n</body></html>\n"
    )
