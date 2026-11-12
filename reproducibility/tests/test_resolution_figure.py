"""Check that Figure 9 remains a faithful display of the saved endpoint."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_srmn2sb2_resolution_fit import (
    MODEL_COLOR, PROFILES, build_figure, plt, resolution_projection,
)


class ResolutionFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        endpoint = ROOT / "data/srmn2sb2_fit_endpoint"
        cls.pred = pd.read_csv(endpoint / "scan_predictions.csv")
        cls.samples = pd.read_csv(endpoint / "resolution_samples.csv.gz")
        cls.modes = pd.read_csv(endpoint / "dispersion_paths.csv")
        cls.figure = build_figure(cls.pred, cls.samples, cls.modes)

    @classmethod
    def tearDownClass(cls):
        plt.close(cls.figure)

    def test_resolution_includes_conditional_energy_variance(self):
        samples = np.array([[-1, -2], [0, 0], [1, 2]], dtype=float)
        expected = np.array([[1, 2], [2, 4.25]])
        np.testing.assert_allclose(resolution_projection(samples, 0.5), expected)
        np.testing.assert_allclose(resolution_projection(samples + [2, 10], 0.5), expected)

    def test_profiles_retain_data_predictions_and_uncertainties(self):
        for ax, filename in zip(self.figure.axes[3:], PROFILES, strict=True):
            group = self.pred[self.pred["filename"] == filename].sort_values("coord")
            data, model, background = ax.lines
            np.testing.assert_array_equal(data.get_xdata(), group["coord"])
            np.testing.assert_array_equal(data.get_ydata(), group["y"] * 1e3)
            np.testing.assert_array_equal(model.get_ydata(), group["prediction"] * 1e3)
            np.testing.assert_array_equal(background.get_ydata(), [group["background"].iloc[0] * 1e3] * 2)
            bars = np.asarray(ax.containers[0].lines[2][0].get_segments())
            np.testing.assert_allclose(bars[:, 0, 1], (group["y"] - group["sigma"]) * 1e3, atol=1e-14)
            np.testing.assert_allclose(bars[:, 1, 1], (group["y"] + group["sigma"]) * 1e3, atol=1e-14)
            self.assertEqual(model.get_color(), MODEL_COLOR)

    def test_panels_have_neutral_titles_and_no_diagnostic_annotations(self):
        self.assertEqual(len(self.figure.axes), 9)
        for letter, ax in zip("abcdefghi", self.figure.axes, strict=True):
            self.assertTrue(ax.get_title(loc="left").startswith(f"({letter})"))
            self.assertEqual(ax._left_title.get_color(), "black")
            self.assertEqual(len(ax.texts), 0)
        self.assertNotIn("r2", self.pred.columns)

    def test_resolution_centers_use_measured_scan_coordinate(self):
        configs = (("SMS_HH_Q002_e10ef14o7m_841228", "H", 0.0875),
                   ("L_Q002_120K_ef28_e10_841965", "L", 2.125))
        for ax, (filename, coord, target) in zip(self.figure.axes[1:3], configs, strict=True):
            points = self.pred[self.pred["filename"] == filename].reset_index(drop=True)
            index = (points["coord"] - target).abs().idxmin()
            point = points.iloc[index]
            cloud = self.samples[(self.samples["filename"] == filename) & (self.samples["point"] == index)]
            self.assertEqual(len(cloud), 64)
            ellipse = ax.patches[0]
            np.testing.assert_array_equal(ellipse.center, [point["coord"], point["E_meV"]])
            angle = np.radians(ellipse.angle)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            actual = rotation @ np.diag([(ellipse.width / 4) ** 2, (ellipse.height / 4) ** 2]) @ rotation.T
            expected = resolution_projection(cloud[[coord, "E_meV"]].to_numpy(), point["sigma_E_conditional"])
            np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
