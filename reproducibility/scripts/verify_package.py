#!/usr/bin/env python3
"""Verify the standalone replay package using only the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    (8, "plot_srmn2sb2_evidence_to_analysis_jeff_review_20261103.py",
     "srmn2sb2_evidence_to_analysis_jeff_review_20261103.pdf"),
    (9, "plot_srmn2sb2_resolution_fit.py", "srmn2sb2_resolution_fit.pdf"),
    (10, "plot_lsmo_live_alignment_chain_jeff_review_20261103.py",
     "lsmo_live_alignment_chain_jeff_review_20261103.pdf"),
    (11, "plot_lsmo_sampling_comparison_jeff_review_20261103.py",
     "lsmo_tasai_sampling_vs_2017_map_jeff_review_20261103.pdf"),
)
IGNORED = {".venv", "outputs", "__pycache__", ".pytest_cache"}
REQUIRED = {
    "README.md", "LICENSE", "requirements.txt", "PROVENANCE.json",
    "scripts/figure_inputs.py", "scripts/verify_package.py", "scripts/replay_figures.py",
    "scripts/build_diagrams.py", "diagrams/figure_sources.json", "DIAGRAMS.md",
    "LSMO_FIT.md", "lsmo_fit_config.json", "results/lsmo_live_fit.json",
    "scripts/replay_lsmo_fit.py", "tests/test_lsmo_fit.py",
    "data/figure_inputs/srmn2sb2_evidence.json",
    "data/figure_inputs/srmn2sb2_dispersion.csv.gz",
    "data/figure_inputs/lsmo_alignment.json", "data/figure_inputs/lsmo_comparison.json",
    "data/srmn2sb2_fit_endpoint/scan_predictions.csv",
    "data/srmn2sb2_fit_endpoint/resolution_samples.csv.gz",
    "data/srmn2sb2_fit_endpoint/dispersion_paths.csv",
    "provenance/source_manifest.json", "provenance/s1_temperature.json",
    "reference_figures/s1/srmn2sb2_order_parameter_lynn.jpg",
    "reference_figures/s1/srmn2sb2_temperature_contour_lynn.jpg",
}
REQUIRED.update("scripts/" + script for _, script, _ in FIGURES)
REQUIRED.update("reference_figures/" + name for _, _, name in FIGURES)
REQUIRED.update("diagrams/" + name for name in (
    "figure_01_triple_axis_schematic.tikz", "figure_02_semantic_decision_stack.tikz",
    "figure_03_federated_architecture.tikz", "figure_04_mcp_tool_layer.tikz",
    "figure_05_claim_lifecycle.tikz", "figure_06_paper_source_linkage.tikz",
    "figure_07_reduction_dag.tikz",
))


def package_files(root: Path) -> set[str]:
    found = set()
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in IGNORED]
        for name in dirs + files:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"Symlink is not allowed: {name}")
        for name in files:
            if name != ".DS_Store":
                found.add((Path(directory) / name).relative_to(root).as_posix())
    return found


def verify(root: Path = ROOT) -> int:
    root = root.resolve()
    listed = set()
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        parts = PurePosixPath(relative)
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or parts.is_absolute()
                or ".." in parts.parts or parts.as_posix() != relative
                or "\\" in relative or relative in listed or relative == "SHA256SUMS"
                or any(part in IGNORED for part in parts.parts)):
            raise ValueError(f"Invalid or duplicate manifest entry: {relative}")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        listed.add(relative)
    if REQUIRED - listed:
        raise ValueError(f"Missing required members: {sorted(REQUIRED - listed)}")
    actual = package_files(root) - {"SHA256SUMS"}
    if actual != listed:
        raise ValueError(f"Package inventory differs: {sorted(actual ^ listed)}")
    return len(listed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(f"Verified {verify(args.root)} packaged files")
