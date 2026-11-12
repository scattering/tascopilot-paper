"""Package integrity, privacy boundary, saved-data invariants, and portable paths."""
from __future__ import annotations

import ast
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figure_inputs
from verify_package import FIGURES, REQUIRED, verify


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def csv_rows(relative):
    data = (ROOT / relative).read_bytes()
    if relative.endswith(".gz"):
        data = gzip.decompress(data)
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


class SavedDataTests(unittest.TestCase):
    def test_package_integrity(self):
        self.assertGreaterEqual(verify(), len(REQUIRED))

    def test_data_preserves_source_hashes_and_declares_adaptations(self):
        original = dict((p, h) for h, p in (
            line.split("  ", 1) for line in (ROOT / "provenance/S2_SHA256SUMS").read_text().splitlines()))
        paths = [p for p in (ROOT / "data").rglob("*") if p.is_file()]
        self.assertEqual(len(paths), 16)
        adaptations = read_json("PROVENANCE.json")["release_data_adaptations"]
        expected_adaptations = {
            "README.md", "RESULTS_stampede_qmc64.md", "dispersion_points_input.csv.gz",
            "endpoint_fit_provenance.json", "export_provenance.json", "result_sha256.txt",
        }
        self.assertEqual({Path(p).name for p in adaptations}, expected_adaptations)
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if relative in adaptations:
                self.assertEqual(adaptations[relative]["source_sha256"], original[relative])
                expected = adaptations[relative]["released_sha256"]
            else:
                expected = original[relative]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_reduced_input_path_migration_preserves_all_other_fields(self):
        rows = csv_rows("data/srmn2sb2_fit_endpoint/dispersion_points_input.csv.gz")
        for row in rows:
            self.assertTrue(row["path"].startswith(("bt7/201804/24858/", "bt7/201806/24858/")))
        values = [{k: v for k, v in row.items() if k != "path"} for row in rows]
        digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, "17d78714e5aa577b7e45b0d00b4f1e6c8d0d3e8789364e0508a91cced8df45d3")

    def test_review_snapshot_contains_only_aggregates(self):
        saved = read_json("data/figure_inputs/srmn2sb2_evidence.json")
        self.assertEqual(set(saved), {"fact_counts", "edge_counts", "findpeaks", "states", "days"})
        for row in saved["fact_counts"]:
            self.assertEqual(set(row), {"paper_id", "type_specific_status", "magnetic", "verdict", "count"})
        self.assertEqual(sum(x["count"] for x in saved["fact_counts"]), 43)
        self.assertEqual(sum(x["count"] for x in saved["fact_counts"] if x["verdict"] == "positive"), 11)
        self.assertEqual(saved["edge_counts"], {"inconclusive": 1, "negative": 5, "positive": 1})
        self.assertEqual(len(saved["days"]), 11)
        self.assertEqual(sum(x["scan_count"] for x in saved["days"]), 409)
        self.assertEqual(sum(len(s["state_epochs"]) for s in saved["states"].values()), 66)
        self.assertEqual(sum(s["summary"]["event_count"] for s in saved["states"].values()), 1573)

    def test_dispersion_count_and_no_private_columns(self):
        rows = csv_rows("data/figure_inputs/srmn2sb2_dispersion.csv.gz")
        self.assertEqual(len(rows), 10522)
        self.assertEqual(set(rows[0]), {"filename", "date", "scan_variable", "H", "K", "L", "E", "Temp", "Monitor", "I_over_M"})

    def test_report_count_matches_original_map_selection(self):
        rows = csv_rows("data/srmn2sb2_fit_endpoint/dispersion_points_input.csv.gz")
        selected = [r for r in rows if float(r["E"]) >= 1 and float(r["Monitor"]) > 0]
        self.assertEqual(len(selected), 10353)
        self.assertEqual(sum(float(r["E"]) == 0 for r in rows), 148)
        self.assertEqual(sum(float(r["E"]) < 0 for r in rows), 12)
        self.assertEqual(sum(0 < float(r["E"]) < 1 for r in rows), 9)
        counts = Counter((r["orient"], r["scan_variable"]) for r in selected)
        summary = csv_rows("provenance/dispersion_map_summary.csv")
        for row in summary:
            self.assertEqual(counts[(row["orient"], row["scan_variable"])], int(row["point_count"]))
        self.assertEqual(sum(int(r["point_count"]) for r in summary), 9152)
        self.assertEqual(sum(r["scan_variable"] == "E" for r in selected), 1201)

    def test_fit_point_sample_join(self):
        points = csv_rows("data/srmn2sb2_fit_endpoint/scan_predictions.csv")
        samples = csv_rows("data/srmn2sb2_fit_endpoint/resolution_samples.csv.gz")
        self.assertEqual(len(points), 799)
        self.assertEqual(len({p["filename"] for p in points}), 20)
        self.assertEqual(len(samples), 799 * 64)
        count = Counter((p["filename"], int(p["point"])) for p in samples)
        expected, per_scan = set(), Counter()
        for point in points:
            name = point["filename"]
            expected.add((name, per_scan[name]))
            per_scan[name] += 1
        self.assertEqual(set(count), expected)
        self.assertEqual(set(count.values()), {64})
        for point in points:
            self.assertGreater(float(point["sigma"]), 0)
            self.assertGreater(float(point["sigma_E_conditional"]), 0)

    def test_endpoint_historical_hashes(self):
        endpoint = ROOT / "data/srmn2sb2_fit_endpoint"
        for line in (endpoint / "result_sha256.txt").read_text().splitlines():
            digest, historical_path = line.split(maxsplit=1)
            path = endpoint / Path(historical_path).name
            data = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
        provenance = read_json("data/srmn2sb2_fit_endpoint/export_provenance.json")
        for key, filename in (("data", "dispersion_points_input.csv.gz"),
                              ("backgrounds", "scan_peak_fits_backgrounds.json"),
                              ("pars", "endpoint_center005_per_scan_qmc64.par")):
            data = (endpoint / filename).read_bytes()
            if filename.endswith(".gz"):
                data = gzip.decompress(data)
            self.assertEqual(hashlib.sha256(data).hexdigest(), provenance["sha256"][key])

    def test_lsmo_inputs_and_phase_origin(self):
        self.assertEqual(len(read_json("data/figure_inputs/lsmo_alignment.json")["scans"]), 6)
        saved = read_json("data/figure_inputs/lsmo_comparison.json")
        self.assertEqual(len(saved["legacy"]), 6)
        self.assertEqual(len(saved["live"]), 40)
        self.assertEqual([p["step"] for p in saved["live"]], list(range(40)))
        self.assertEqual(Counter(p["surrogate"] for p in saved["live"]),
                         {"space_filling": 10, "tasai_log_gp": 15, "tasai_fm_andie": 15})
        self.assertEqual(saved["phase_provenance"], ["archive filename and step thresholds"])
        for point in saved["live"]:
            self.assertGreater(point["sigma"], 0)

    def test_plot_scripts_match_sources_except_recorded_heading(self):
        provenance = read_json("PROVENANCE.json")
        sources = provenance["copied_sources"]
        adapted = provenance["release_figure_adaptations"]
        self.assertEqual(set(adapted), {"scripts/plot_lsmo_live_alignment_chain_jeff_review_20261103.py"})
        for _, name, _ in FIGURES:
            path = ROOT / "scripts" / name
            key = "scripts/" + name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if key in adapted:
                # One display string differs from the source script; restoring it must reproduce the source hash.
                self.assertEqual(adapted[key]["source_sha256"], sources[key]["sha256"])
                self.assertEqual(digest, adapted[key]["released_sha256"])
                restored = path.read_bytes().replace(b"TASCopilot as the orchestrator", b"TAS Co-pilot as the orchestrator")
                self.assertEqual(hashlib.sha256(restored).hexdigest(), sources[key]["sha256"])
            else:
                self.assertEqual(digest, sources[key]["sha256"])
            tree = ast.parse(path.read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.split(".")[0])
            self.assertNotIn("tas_copilot_core", imports)
            self.assertNotIn("tasreduction", imports)
            for token in ("Path.home()", "/Users/", "/home/", "/work2/"):
                self.assertNotIn(token, path.read_text())
        text = (ROOT / "scripts/plot_lsmo_sampling_comparison_jeff_review_20261103.py").read_text()
        self.assertIn("same monitor normalization; no rescaling", text)

    def test_s1_assets_and_sanitized_provenance(self):
        record = read_json("provenance/s1_temperature.json")
        self.assertEqual(record["sample_and_transition"]["reported_neel_temperature"], "114.6(6) K")
        self.assertEqual(record["sample_and_transition"]["reported_beta"], 0.21)
        self.assertEqual(record["sample_and_transition"]["mass_g"], 0.5)
        self.assertFalse(record["availability"]["numerical_panel_source_data_included"])
        self.assertFalse(record["availability"]["panel_fit_routine_included"])
        self.assertFalse(record["availability"]["panel_refit_performed"])
        self.assertNotIn("gmail_message_id", json.dumps(record))
        self.assertNotIn("email_subject", json.dumps(record))
        for asset in record["assets"]:
            self.assertEqual(hashlib.sha256((ROOT / asset["path"]).read_bytes()).hexdigest(), asset["sha256"])


class PathTests(unittest.TestCase):
    def test_defaults_are_package_relative(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work).resolve()
            with patch.object(figure_inputs, "__file__", str(root / "scripts/figure_inputs.py")), patch("sys.argv", ["plot.py"]):
                args = figure_inputs.plot_args("figure.pdf")
            self.assertEqual(args.data_dir, root / "data")
            self.assertEqual(args.out, root / "outputs/figure.pdf")

    def test_explicit_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            with patch("sys.argv", ["plot.py", "--data-dir", str(root / "external"), "--out", str(root / "custom/f.pdf")]):
                args = figure_inputs.plot_args("default.pdf")
            self.assertEqual(args.data_dir, root / "external")
            self.assertEqual(args.out, root / "custom/f.pdf")

    def test_missing_json_diagnostic(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing packaged input"):
            figure_inputs.read_json(ROOT / "does-not-exist.json")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "member.txt").write_text("fixture")
        self.digest = hashlib.sha256(b"fixture").hexdigest()
        (self.root / "SHA256SUMS").write_text(f"{self.digest}  member.txt\n")
        self.required = patch("verify_package.REQUIRED", {"member.txt"})
        self.required.start()
        self.addCleanup(self.required.stop)

    def test_valid(self):
        self.assertEqual(verify(self.root), 1)

    def test_modified_file_rejected(self):
        (self.root / "member.txt").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            verify(self.root)

    def test_missing_required_rejected(self):
        with patch("verify_package.REQUIRED", {"missing.txt"}):
            with self.assertRaisesRegex(ValueError, "Missing required"):
                verify(self.root)

    def test_unlisted_file_rejected(self):
        (self.root / "unlisted.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            verify(self.root)

    def test_duplicate_rejected(self):
        (self.root / "SHA256SUMS").write_text(f"{self.digest}  member.txt\n" * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify(self.root)

    def test_traversal_rejected(self):
        (self.root / "SHA256SUMS").write_text(f"{self.digest}  ../member.txt\n")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            verify(self.root)

    def test_symlink_rejected(self):
        (self.root / "link").symlink_to(self.root / "member.txt")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
