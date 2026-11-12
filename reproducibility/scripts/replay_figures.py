#!/usr/bin/env python3
"""Replay Figures 8-11 outside the checkout; optionally compare PDF raster pixels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from verify_package import FIGURES, ROOT, verify


def compare_pdf(actual: Path, reference: Path) -> dict:
    from PIL import Image, ImageChops

    def render(path):
        result = subprocess.run(
            ["pdftoppm", "-singlefile", "-r", "120", "-png", str(path)],
            check=True, capture_output=True,
        )
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")

    left, right = render(actual), render(reference)
    equal = left.size == right.size and ImageChops.difference(left, right).getbbox() is None
    return {"pixel_identical": equal, "generated_size": list(left.size),
            "reference_size": list(right.size), "render_dpi": 120}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/replay",
                        help="New output directory; existing paths are rejected")
    parser.add_argument("--compare-reference", action="store_true",
                        help="Require pixel-identical 120-dpi renders against Overleaf figures; needs Poppler")
    args = parser.parse_args()
    count = verify()
    if args.compare_reference and not shutil.which("pdftoppm"):
        parser.error("--compare-reference requires pdftoppm from Poppler")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "Pillow")},
              "package_files_verified": count, "empty_home": True,
              "unrelated_working_directory": True, "no_tas_packages": False, "figures": []}
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as cwd:
        env = {k: v for k, v in os.environ.items() if k not in
               {"HOME", "PYTHONHOME", "PYTHONPATH", "MPLCONFIGDIR"}}
        env.update(HOME=home, MPLCONFIGDIR=str(Path(home) / "mpl"),
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
        subprocess.run([sys.executable, "-B", "-c",
                        "import importlib.util; "
                        "assert importlib.util.find_spec('tas_copilot_core') is None, 'Use a clean venv'; "
                        "assert importlib.util.find_spec('tasreduction') is None, 'Use a clean venv'"],
                       check=True, cwd=cwd, env=env)
        result["no_tas_packages"] = True
        for number, script, filename in FIGURES:
            output = out / filename
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script),
                            "--data-dir", str(ROOT / "data"), "--out", str(output)],
                           check=True, cwd=cwd, env=env)
            png = output.with_suffix(".png")
            if not output.read_bytes().startswith(b"%PDF-") or not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Invalid outputs for Figure {number}")
            entry = {"figure": number, "pdf": filename, "png": png.name,
                     "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                     "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()}
            if args.compare_reference:
                entry["reference_comparison"] = compare_pdf(output, ROOT / "reference_figures" / filename)
            result["figures"].append(entry)
    result["status"] = "passed" if all(
        item.get("reference_comparison", {}).get("pixel_identical", True)
        for item in result["figures"]
    ) else "render_mismatch"
    (out / "replay_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit("Reference render differs; inspect the result, do not silently replace saved figures")


if __name__ == "__main__":
    main()
