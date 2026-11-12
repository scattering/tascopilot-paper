"""Shared command-line paths for the paper's portable figure scripts."""
import argparse
import json
from pathlib import Path


def plot_args(filename):
    root = Path(__file__).resolve().parents[1]
    default_data = root / "data"
    parser = argparse.ArgumentParser(description="Replot saved results; no reduction or spin-wave refit.")
    parser.add_argument("--data-dir", type=Path, default=default_data,
                        help="S2 data directory, containing fit-endpoint and figure-input subdirectories")
    parser.add_argument("--out", type=Path, default=root / "outputs" / filename)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    return args


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged input: {path}. Pass --data-dir PATH/TO/reproducibility/data.")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
