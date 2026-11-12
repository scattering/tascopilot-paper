#!/usr/bin/env python3
"""Refit the 40 saved monitor-normalized LSMO points used for Table 4."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lsmo_fit_config.json"
REFERENCE = ROOT / "results/lsmo_live_fit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_observations(path: Path) -> list[tuple[float, ...]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["live"]
    if len(rows) != 40 or [r["step"] for r in rows] != list(range(40)):
        raise ValueError("Expected all 40 live points in acquisition order")
    observations = []
    for row in rows:
        dq, energy, intensity, sigma = (float(row[k]) for k in ("dq", "E", "I", "sigma"))
        if not all(math.isfinite(v) for v in (dq, energy, intensity, sigma)) or sigma <= 0:
            raise ValueError(f"Invalid observation at step {row['step']}")
        observations.append((0.0, 1.0 + dq, 0.0, energy, intensity, sigma))
    return observations


def run_fit(tasai_root: Path | None = None) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ROOT / config["input"]
    fitter = ROOT.parent / config["fitter"]
    for path, expected in ((source, config["input_sha256"]),
                           (fitter, config["fitter_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Input or model checksum mismatch: {path.name}")
    if tasai_root is not None:
        tasai_root = tasai_root.resolve()
        if not (tasai_root / "tasai/physics/base.py").is_file():
            raise ValueError("--tasai-root must be a TAS-AI source checkout")
        sys.path.insert(0, str(tasai_root))
    # Import before the legacy fitter can attempt author-machine discovery.
    try:
        base = importlib.import_module("tasai.physics.base")
    except ImportError as exc:
        raise RuntimeError("Install the pinned TAS-AI source or pass --tasai-root; see LSMO_FIT.md") from exc
    if sha256(Path(base.__file__)) != config["tasai"]["base_sha256"]:
        raise ValueError("TAS-AI physics base differs from the pinned replay source")
    os.environ["TASAI_PATH"] = str(Path(base.__file__).resolve().parents[2])
    sys.path.insert(0, str(fitter.parent))
    from tasai_fm_magnon import FMMagnonModel, fit_hypotheses
    import numpy as np
    import scipy

    observations = load_observations(source)
    models, _, diagnostics = fit_hypotheses(
        observations, lattice_a=config["lattice_a_angstrom"],
        T=config["temperature_K"], kinds=(config["kind"],))
    model, diagnostic = models[0], diagnostics[0]
    if diagnostic["fit_error"]:
        raise RuntimeError(diagnostic["fit_error"])
    predictions = [float(model.compute_intensity(*row[:4])) for row in observations]
    residuals = [(pred - row[4]) / row[5] for pred, row in zip(predictions, observations)]
    chi2 = sum(r * r for r in residuals)
    if not math.isfinite(chi2) or not math.isclose(chi2, diagnostic["chi2"], rel_tol=1e-7):
        raise RuntimeError("Returned parameters do not reproduce the fit objective")
    bounds = {p.name: list(p.bounds) for p in model.free_parameters}
    bound_hits = {p.name: side for p in model.free_parameters
                  for side, value in zip(("lower", "upper"), p.bounds)
                  if math.isclose(p.value, value, rel_tol=0, abs_tol=1e-5)}
    initial = FMMagnonModel(config["kind"])
    return {
        "schema_version": 1,
        "config": config,
        "config_sha256": sha256(CONFIG),
        "replay_script_sha256": sha256(Path(__file__)),
        "command": "python reproducibility/scripts/replay_lsmo_fit.py --tasai-root /path/to/tasai --check",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "method": "scipy.optimize.least_squares; released fit_hypotheses; default initialization; max_nfev=2000",
        "initial_parameters": {p.name: float(p.value) for p in initial.parameters},
        "initial_scale": "amplitude and background initialized from the observations by fit_model",
        "bounds": bounds,
        "bound_hits": bound_hits,
        "n_points": len(observations),
        "n_free": diagnostic["n_free"],
        "degrees_of_freedom": len(observations) - diagnostic["n_free"],
        "chi2": chi2,
        "chi2_per_dof": chi2 / (len(observations) - diagnostic["n_free"]),
        "parameters": diagnostic["params"],
        "predictions": predictions,
        "standardized_residuals": residuals,
    }


def compare(result: dict, expected: dict) -> None:
    for key in ("config_sha256", "config", "bound_hits", "n_points", "n_free", "degrees_of_freedom", "bounds"):
        if result[key] != expected[key]:
            raise ValueError(f"Replay metadata differs: {key}")
    checks = [("chi2", result["chi2"], expected["chi2"])]
    checks.extend((name, value, expected["parameters"][name])
                  for name, value in result["parameters"].items())
    if len(result["predictions"]) != len(expected["predictions"]):
        raise ValueError("Prediction count differs")
    checks.extend((f"prediction {i}", value, target) for i, (value, target)
                  in enumerate(zip(result["predictions"], expected["predictions"])))
    for name, value, target in checks:
        if not math.isclose(value, target, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Replay differs for {name}: {value} versus {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasai-root", type=Path, help="Pinned external TAS-AI checkout")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/lsmo_fit/replay_result.json")
    parser.add_argument("--check", action="store_true", help="Compare all parameters and predictions with the saved result")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new --out path")
    try:
        result = run_fit(args.tasai_root)
        if args.check:
            compare(result, json.loads(REFERENCE.read_text(encoding="utf-8")))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    p = result["parameters"]
    print(f"40 points: D={p['D']:.4f} meV A^2, beta={p['beta']:.4f} A^2, "
          f"chi2/dof={result['chi2_per_dof']:.4f}")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
