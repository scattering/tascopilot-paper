#!/usr/bin/env python3
"""Build figure-only PDFs for Figures 1-7 from preserved TikZ graphic blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from verify_package import ROOT, verify

SOURCE = ROOT / "diagrams/figure_sources.json"
REVISION = "a9173895c687129310797ec93b44e6e7b5f89dc0"
LIBRARIES = ["arrows.meta", "positioning", "shapes.geometric", "angles", "quotes", "calc"]
FONT_SIZES = {"normalsize": [9, 11.5], "footnotesize": [7, 9], "scriptsize": [9, 9]}
PACKAGES = ["standalone.cls", "fontenc.sty", "stix2.sty", "anyfontsize.sty", "xcolor.sty", "tikz.sty"]
GRAPHIC_COMMANDS = {
    "begin", "end", "coordinate", "draw", "node", "pic", "scriptsize",
    "footnotesize", "itshape", "color", "mathbf", "mathrm", "theta",
}


def validate_sources(record: dict) -> None:
    if record["schema_version"] != 1 or record["source_revision"] != REVISION:
        raise ValueError("Unexpected diagram source schema or revision")
    style = record["style"]
    if (style["tikz_libraries"] != LIBRARIES or style["font_sizes_pt"] != FONT_SIZES
            or style["border_pt"] != 2
            or style["packages"] != ["fontenc[T1]", "stix2", "anyfontsize", "xcolor", "tikz"]):
        raise ValueError("Unexpected diagram rendering settings")
    if [item["figure"] for item in record["figures"]] != list(range(1, 8)):
        raise ValueError("Expected exactly Figures 1-7 in order")
    for item in record["figures"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]+", item["name"]):
            raise ValueError("Unsafe diagram output name")
        expected = f"diagrams/figure_{item['figure']:02d}_{item['name']}.tikz"
        if item["graphic_file"] != expected:
            raise ValueError("Unexpected graphic source path")
        code = item["graphic_code"]
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != item["graphic_sha256"]:
            raise ValueError(f"Graphic checksum mismatch: Figure {item['figure']}")
        if (not code.startswith(r"\begin{tikzpicture}[") or not code.endswith("\\end{tikzpicture}\n")
                or code.count(r"\begin{tikzpicture}") != 1 or code.count(r"\end{tikzpicture}") != 1):
            raise ValueError("Expected one bounded TikZ graphic and no surrounding text")
        if re.findall(r"\\(?:begin|end)\{([^}]+)\}", code) != ["tikzpicture", "tikzpicture"]:
            raise ValueError("Only the TikZ environment is allowed")
        # This is a regression guard for the audited code, not a general TeX sandbox.
        commands = {token for token in re.findall(r"\\([A-Za-z@]+|.)", code)
                    if re.fullmatch(r"[A-Za-z@]+", token)}
        if commands - GRAPHIC_COMMANDS:
            raise ValueError(f"Unexpected non-graphic TeX commands: {sorted(commands - GRAPHIC_COMMANDS)}")


def load_sources(path: Path = SOURCE) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    for item in record["figures"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]+", item["name"]):
            raise ValueError("Unsafe diagram output name")
        expected = f"diagrams/figure_{item['figure']:02d}_{item['name']}.tikz"
        if item["graphic_file"] != expected:
            raise ValueError("Unexpected graphic source path")
        item["graphic_code"] = (path.parent / Path(expected).name).read_text(encoding="utf-8")
    validate_sources(record)
    return record


def document(graphic: dict, style: dict) -> str:
    fonts = "\n".join(
        rf"\renewcommand{{\{name}}}{{\fontsize{{{size}}}{{{baseline}}}\selectfont}}"
        for name, (size, baseline) in style["font_sizes_pt"].items()
    )
    return (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage[T1]{fontenc}\n\\usepackage{stix2}\n"
        "\\usepackage{anyfontsize,xcolor,tikz}\n"
        "\\usetikzlibrary{" + ",".join(style["tikz_libraries"]) + "}\n"
        + fonts + "\n\\begin{document}\n\\normalsize\n"
        + graphic["graphic_code"] + "\\end{document}\n"
    )


def dependencies(render: bool = False) -> dict:
    tools = ["pdflatex", "kpsewhich"] + (["pdftoppm"] if render else [])
    missing = [name for name in tools if not shutil.which(name)]
    packages = {}
    if shutil.which("kpsewhich"):
        for name in PACKAGES:
            found = subprocess.run(["kpsewhich", name], capture_output=True, text=True, check=False)
            path = Path(found.stdout.strip())
            if found.returncode or not path.is_file():
                missing.append(name)
            else:
                packages[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"status": "not_available" if missing else "available", "missing": missing,
            "package_sha256": packages}


def build(record: dict, out: Path, render: bool = False) -> dict:
    validate_sources(record)
    available = dependencies(render)
    if available["status"] != "available":
        return available
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "passed", "source_revision": record["source_revision"],
              "shell_escape": False, "dependencies": available,
              "pdflatex_version": subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0],
              "figures": []}
    with tempfile.TemporaryDirectory() as home:
        env = {k: v for k, v in os.environ.items()
               if k not in {"TEXINPUTS", "TEXMFOUTPUT", "TEXMFHOME", "TEXMFVAR", "TEXMFCACHE", "HOME"}}
        env.update(HOME=home, TEXMFHOME=str(Path(home) / "texmf"),
                   TEXMFVAR=str(Path(home) / "texmf-var"), openin_any="p", openout_any="p")
        for item in record["figures"]:
            stem = f"figure_{item['figure']:02d}_{item['name']}"
            tex = out / f"{stem}.tex"
            tex.write_text(document(item, record["style"]), encoding="utf-8")
            completed = subprocess.run(
                ["pdflatex", "-no-shell-escape", "-halt-on-error", "-interaction=nonstopmode", tex.name],
                cwd=out, env=env, capture_output=True, text=True, timeout=120,
            )
            (out / f"{stem}.build.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"Figure {item['figure']} failed; inspect {stem}.build.log")
            pdf = tex.with_suffix(".pdf")
            if not pdf.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"No PDF produced for Figure {item['figure']}")
            if not re.search(r"Output written on [\s\S]*?\(1 page,", completed.stdout):
                raise ValueError(f"Expected one figure-only page for Figure {item['figure']}")
            entry = {"figure": item["figure"], "pdf": pdf.name,
                     "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                     "graphic_sha256": item["graphic_sha256"]}
            if render:
                subprocess.run(["pdftoppm", "-singlefile", "-r", "120", "-png", pdf.name, stem],
                               cwd=out, check=True, capture_output=True, timeout=120)
                png = tex.with_suffix(".png")
                entry.update(png=png.name, png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(), render_dpi=120)
            result["figures"].append(entry)
    (out / "diagram_build_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/diagrams",
                        help="New output directory for figure-only PDFs and generated TeX/logs")
    parser.add_argument("--render", action="store_true", help="Also render each PDF to PNG with Poppler")
    parser.add_argument("--check-dependencies", action="store_true", help="Report availability without writing output")
    args = parser.parse_args()
    verify()
    record = load_sources()
    result = dependencies(args.render) if args.check_dependencies else build(record, args.out, args.render)
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "not_available" else 0


if __name__ == "__main__":
    raise SystemExit(main())
