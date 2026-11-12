#!/usr/bin/env python3
"""Check the staged code/data release without printing sensitive file content."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {".git", ".claude", ".venv", "venv", "__pycache__", "local_runs", "tmp", "notes"}
FORBIDDEN_SUFFIXES = {".tex", ".bib", ".doc", ".docx", ".sqlite", ".sqlite3", ".db", ".zip", ".tar"}
SECRET_PATTERNS = (
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    rb"\bAKIA[0-9A-Z]{16}\b",
    rb"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,}\b",
    rb"https?://[^\s/:]+:[^\s/@]+@",
)


def check_content(
    name: str, content: bytes, synthetic_catalog_hashes: dict[str, str] | None = None
) -> list[str]:
    path = PurePosixPath(name)
    problems = []
    if path.is_absolute() or ".." in path.parts:
        problems.append("unsafe path")
    if any(part in FORBIDDEN_DIRS for part in path.parts):
        problems.append("excluded directory")
    if path.name == ".webui_secret_key" or "handoff" in path.name.lower() or (
        path.name.startswith(".env") and not path.name.endswith(".example")
    ):
        problems.append("private configuration or handoff")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        problems.append("manuscript, database, or opaque archive")
    if path.suffix.lower() == ".pdf" and path.parent != PurePosixPath("reproducibility/reference_figures"):
        problems.append("PDF outside figure-only references")
    if str(path).startswith("data/catalogs/"):
        allowed = (synthetic_catalog_hashes or {}).get(name)
        if allowed != hashlib.sha256(content).hexdigest():
            problems.append("catalog is not an approved synthetic release fixture")
    if path.suffix == ".gz":
        if not name.endswith(".csv.gz"):
            problems.append("unexpected compressed content")
        else:
            try:
                content = gzip.decompress(content)
            except (OSError, EOFError):
                problems.append("invalid gzip data")
    if any(re.search(pattern, content) for pattern in SECRET_PATTERNS):
        problems.append("credential pattern (content redacted)")
    if re.search(rb"/Users/" + rb"(?!someone/)[^/\s\"']+/", content) or re.search(rb"/work2/" + rb"\d+/[^/\s]+/", content):
        problems.append("personal filesystem path")
    return problems


def staged_files() -> list[tuple[str, bytes]]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"], cwd=ROOT,
        check=True, capture_output=True,
    )
    files = []
    for row in result.stdout.split(b"\0"):
        if not row:
            continue
        meta, path = row.split(b"\t", 1)
        mode, oid, stage = meta.split()
        name = path.decode("utf-8")
        if mode not in {b"100644", b"100755"} or stage != b"0":
            raise ValueError(f"Unsupported staged entry: {name}")
        content = subprocess.run(
            ["git", "cat-file", "blob", oid.decode()], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout
        files.append((name, content))
    if not files:
        raise ValueError("Stage the intended release files before auditing")
    return files


def audit() -> dict:
    files = staged_files()
    by_name = dict(files)
    provenance = json.loads(by_name.get("RELEASE_PROVENANCE.json", b"{}"))
    synthetic_catalog_hashes = provenance.get("synthetic_catalog_sha256", {})
    findings = [
        {"path": name, "issues": issues}
        for name, data in files
        if (issues := check_content(name, data, synthetic_catalog_hashes))
    ]
    expected = {
        row["path"]: row["sha256"]
        for row in provenance.get("source_software", {}).get("files", [])
    }
    for line in by_name.get("reproducibility/SHA256SUMS", b"").decode().splitlines():
        digest, relative = line.split("  ", 1)
        expected[f"reproducibility/{relative}"] = digest
    for name, digest in expected.items():
        if name not in by_name or hashlib.sha256(by_name[name]).hexdigest() != digest:
            findings.append({"path": name, "issues": ["release checksum mismatch or missing file"]})
    return {
        "status": "failed" if findings else "passed",
        "scope": "Git index; content patterns also inspect compressed CSVs",
        "file_count": len(files),
        "bytes": sum(len(data) for _, data in files),
        "manifest_files_verified": len(expected),
        "findings": findings,
        "files": [
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in files
        ],
        "note": "Pattern checks supplement, not replace, review of the explicit export inventory.",
    }


class AuditTests(unittest.TestCase):
    def test_rejects_personal_paths_and_any_handoff_name(self):
        personal = b"/Users/" + b"example-person/private/input.csv"
        self.assertTrue(check_content("data/input.csv", personal))
        self.assertTrue(check_content("data/input.csv.gz", gzip.compress(personal)))
        self.assertTrue(check_content("docs/RELEASE_HANDOFF_2026.md", b"internal"))

    def test_rejects_manuscript_and_credentials(self):
        for path in ("main.tex", "paper.pdf", "cover.docx", ".webui_secret_key", ".env"):
            self.assertTrue(check_content(path, b"test"), path)

    def test_allows_only_figure_pdf_location(self):
        self.assertFalse(check_content("reproducibility/reference_figures/figure.pdf", b"figure"))
        self.assertTrue(check_content("docs/figure.pdf", b"figure"))

    def test_scans_compressed_data(self):
        fake_key = b"gh" + b"p_" + b"X" * 36
        self.assertTrue(check_content("data/example.csv.gz", gzip.compress(fake_key)))
        self.assertFalse(check_content("data/example.csv.gz", gzip.compress(b"x,y\n1,2\n")))

    def test_rejects_unsafe_and_private_paths(self):
        for path in ("../outside", "/outside", "local_runs/test.json", "data/catalogs/papers.jsonl"):
            self.assertTrue(check_content(path, b"test"), path)

    def test_synthetic_catalog_allowlist_checks_bytes(self):
        path, content = "data/catalogs/experiments.jsonl", b"synthetic example"
        approved = {path: hashlib.sha256(content).hexdigest()}
        self.assertFalse(check_content(path, content, approved))
        self.assertTrue(check_content(path, content + b"changed", approved))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON inventory outside the release")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    report = audit()
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(ROOT):
            parser.error("Write audit inventories outside the repository to avoid recursive manifests")
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
