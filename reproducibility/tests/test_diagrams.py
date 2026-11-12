"""Keep diagram source bounded to graphics and independent of manuscript files."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_diagrams import FONT_SIZES, REVISION, build, document, load_sources, validate_sources
from verify_package import package_files


class DiagramTests(unittest.TestCase):
    def setUp(self):
        self.record = load_sources()

    def test_all_seven_figures_have_provenance(self):
        self.assertEqual(self.record["source_revision"], REVISION)
        self.assertEqual([r["figure"] for r in self.record["figures"]], list(range(1, 8)))
        for row in self.record["figures"]:
            self.assertEqual(len(row["source_file_sha256"]), 64)
            start, end = row["source_lines"]
            self.assertEqual(end - start + 1, len(row["graphic_code"].splitlines()))
        self.assertEqual(self.record["figures"][0]["source_file"], "figures/triple_axis_schematic_jeff_review_20261103.tex")
        self.assertEqual([r["source_file"] for r in self.record["figures"][-2:]], ["sections/03_workflows.tex"] * 2)

    def test_plain_graphic_sources_and_metadata_separate(self):
        metadata = json.loads((ROOT / "diagrams/figure_sources.json").read_text())
        for row in metadata["figures"]:
            self.assertNotIn("graphic_code", row)
            self.assertTrue(row["graphic_file"].endswith(".tikz"))
            self.assertTrue((ROOT / row["graphic_file"]).is_file())

    def test_no_manuscript_environment_or_caption(self):
        for row in self.record["figures"]:
            for token in (r"\caption", r"\label", r"\cite", r"\section", r"\bibliography",
                          r"\input", r"\include", r"\documentclass", r"\begin{figure", r"\end{figure"):
                self.assertNotIn(token, row["graphic_code"])

    def test_source_modification_rejected(self):
        self.record["figures"][0]["graphic_code"] += "extra"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_sources(self.record)

    def test_caption_rejected_even_with_changed_hash(self):
        row = self.record["figures"][0]
        row["graphic_code"] = row["graphic_code"].replace(r"\end{tikzpicture}", r"\caption{not a graphic}\end{tikzpicture}")
        row["graphic_sha256"] = hashlib.sha256(row["graphic_code"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "non-graphic"):
            validate_sources(self.record)

    def test_external_read_rejected(self):
        row = self.record["figures"][0]
        row["graphic_code"] = row["graphic_code"].replace(r"\end{tikzpicture}", r"\input{private}\end{tikzpicture}")
        row["graphic_sha256"] = hashlib.sha256(row["graphic_code"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "non-graphic"):
            validate_sources(self.record)

    def test_unsafe_name_rejected(self):
        self.record["figures"][0]["name"] = "../../elsewhere"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            validate_sources(self.record)

    def test_missing_figure_rejected(self):
        self.record["figures"].pop()
        with self.assertRaisesRegex(ValueError, "exactly Figures"):
            validate_sources(self.record)

    def test_font_sizes_and_exact_graphics_retained(self):
        self.assertEqual(self.record["style"]["font_sizes_pt"], FONT_SIZES)
        for row in self.record["figures"]:
            generated = document(row, self.record["style"])
            self.assertIn(row["graphic_code"], generated)
            self.assertIn(r"\usepackage{stix2}", generated)
            self.assertNotIn("USG", generated)
            self.assertNotIn(r"\caption", generated)

    def test_no_tex_or_manuscript_class_distributed(self):
        files = package_files(ROOT)
        self.assertFalse(any(Path(p).suffix in {".tex", ".cls", ".bib", ".bbl"} for p in files))

    def test_missing_tex_returns_unavailable_without_output(self):
        with tempfile.TemporaryDirectory() as work:
            out = Path(work) / "not-created"
            with patch("build_diagrams.dependencies", return_value={"status": "not_available", "missing": ["pdflatex"]}):
                result = build(self.record, out)
            self.assertEqual(result["status"], "not_available")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
