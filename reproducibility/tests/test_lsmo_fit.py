"""Saved LSMO fit invariants; optimizer replay is the documented --check CLI."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_lsmo_fit import compare, load_observations, sha256


class LSMOFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "lsmo_fit_config.json").read_text())
        cls.result = json.loads((ROOT / "results/lsmo_live_fit.json").read_text())
        cls.source = ROOT / cls.config["input"]

    def test_exact_input_model_and_replay_hashes(self):
        self.assertEqual(sha256(self.source), self.config["input_sha256"])
        self.assertEqual(sha256(ROOT.parent / self.config["fitter"]), self.config["fitter_sha256"])
        self.assertEqual(self.result["config"], self.config)
        self.assertEqual(sha256(ROOT / "lsmo_fit_config.json"), self.result["config_sha256"])
        self.assertEqual(sha256(ROOT / "scripts/replay_lsmo_fit.py"), self.result["replay_script_sha256"])

    def test_all_points_and_normalization(self):
        rows = json.loads(self.source.read_text())["live"]
        observations = load_observations(self.source)
        self.assertEqual(len(observations), 40)
        for r, obs in zip(rows, observations):
            self.assertEqual(obs, (0.0, 1.0 + r["dq"], 0.0, r["E"], r["I"], r["sigma"]))
        self.assertEqual(self.config["normalization"], "counts per 150000 incident-beam monitor counts")
        self.assertEqual(self.config["lattice_a_angstrom"], 3.9005)
        self.assertFalse(self.config["higher_order_monitor_correction"])
        self.assertFalse(self.config["resolution_convolution"])

    def test_rejects_missing_or_reordered_points(self):
        saved = json.loads(self.source.read_text())
        for rows in (saved["live"][:-1], saved["live"][::-1]):
            with patch.object(Path, "read_text", return_value=json.dumps({"live": rows})):
                with self.assertRaisesRegex(ValueError, "40 live points"):
                    load_observations(self.source)

    def test_rejects_invalid_uncertainty(self):
        for sigma in (0, -1, float("nan")):
            saved = json.loads(self.source.read_text())
            saved["live"][0]["sigma"] = sigma
            with patch.object(Path, "read_text", return_value=json.dumps(saved)):
                with self.assertRaisesRegex(ValueError, "Invalid observation"):
                    load_observations(self.source)

    def test_result_arithmetic_and_paper_precision(self):
        result = self.result
        self.assertEqual(result["n_points"], 40)
        self.assertEqual(result["n_free"], 5)
        self.assertEqual(result["degrees_of_freedom"], 35)
        self.assertEqual(len(result["predictions"]), 40)
        self.assertEqual(len(result["standardized_residuals"]), 40)
        for row, pred, resid in zip(load_observations(self.source), result["predictions"],
                                    result["standardized_residuals"]):
            self.assertAlmostEqual((pred - row[4]) / row[5], resid)
        self.assertAlmostEqual(sum(r*r for r in result["standardized_residuals"]), result["chi2"])
        self.assertAlmostEqual(result["chi2"] / 35, result["chi2_per_dof"])
        self.assertEqual(f"{result['parameters']['D']:.1f}", "131.9")
        self.assertEqual(f"{result['parameters']['beta']:.2f}", "2.00")
        self.assertEqual(round(result["chi2_per_dof"]), 149)
        self.assertEqual(result["bound_hits"], {"beta": "upper"})
        self.assertEqual(result["bounds"]["beta"], [0.0, 2.0])
        for name, value in result["parameters"].items():
            lo, hi = result["bounds"][name]
            self.assertTrue(math.isfinite(value) and lo <= value <= hi)

    def test_comparison_detects_changed_fit_and_predictions(self):
        compare(self.result, self.result)
        for field in ("parameters", "predictions", "config"):
            changed = copy.deepcopy(self.result)
            if field == "parameters":
                changed[field]["D"] = 135.3
            elif field == "predictions":
                changed[field][-1] += 1
            else:
                changed[field]["normalization"] = "counts per second"
            with self.assertRaises(ValueError):
                compare(changed, self.result)


if __name__ == "__main__":
    unittest.main()
