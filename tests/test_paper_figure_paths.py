"""Manuscript replay defaults must not write artifacts into source directories."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "reproducibility/scripts/figure_inputs.py"
SPEC = importlib.util.spec_from_file_location("paper_figure_inputs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PaperFigurePathsTest(unittest.TestCase):
    def test_defaults_use_ignored_repository_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve() / "repository"
            helper = repo / "reproducibility/scripts/figure_inputs.py"
            with patch.object(MODULE, "__file__", str(helper)), patch(
                "sys.argv", ["plot.py"]
            ):
                args = MODULE.plot_args("figure.pdf")
            work = repo / "reproducibility"
            self.assertEqual(args.data_dir, work / "data")
            self.assertEqual(args.out, work / "outputs/figure.pdf")
            self.assertTrue(args.out.parent.is_dir())
            self.assertFalse((repo / "scripts/images").exists())

    def test_explicit_external_inputs_and_output_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "external/data"
            output = root / "custom_output/figure.pdf"
            with patch("sys.argv", ["plot.py", "--data-dir", str(data), "--out", str(output)]):
                args = MODULE.plot_args("default.pdf")
            self.assertEqual(args.data_dir, data)
            self.assertEqual(args.out, output)
            self.assertTrue(output.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
