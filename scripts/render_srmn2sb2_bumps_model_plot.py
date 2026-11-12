#!/usr/bin/env python3
"""Render the native Bumps model/data panel at an existing SrMn2Sb2 .par.

This is a plotting-only adapter.  It loads the ordinary Bumps ``FitProblem``,
requires a complete parameter file, and delegates rendering to
``FitProblem.plot`` so the output follows the same path used by Bumps after a
fit.  It does not optimize, resample, or alter parameter constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--pars", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--seed", choices=("all102", "center005"), required=True)
    parser.add_argument(
        "--normalization", choices=("per_scan", "shared_zone_config"), required=True
    )
    parser.add_argument("--domain", choices=("single_y", "equal3"), default="single_y")
    parser.add_argument(
        "--resolution", choices=("legacy", "header_horizontal"), default="legacy"
    )
    parser.add_argument("--qmc", type=int, default=64)
    parser.add_argument(
        "--scheduling",
        choices=("official_processpool", "outer_thread1"),
        default="official_processpool",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from bumps.cli import load_best, load_model

    paths = (args.model, args.data, args.backgrounds, args.pars)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.qmc <= 0 or args.qmc & (args.qmc - 1):
        raise ValueError("--qmc must be a positive power of two")

    model_options = [
        str(args.data.resolve()),
        str(args.backgrounds.resolve()),
        args.seed,
        args.normalization,
        args.domain,
        args.resolution,
        str(args.qmc),
        "full",
        args.scheduling,
    ]
    problem = load_model(str(args.model.resolve()), model_options=model_options)
    load_best(problem, args.pars.resolve())
    undefined = np.asarray(getattr(problem, "undefined", []), dtype=bool)
    if undefined.size and np.any(undefined):
        missing = np.asarray(problem.labels(), dtype=object)[undefined].tolist()
        raise ValueError(f"parameter file is incomplete; missing {missing}")

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    nllf = float(problem.nllf())
    if not np.isfinite(nllf):
        raise ValueError("non-finite objective at plotted parameter point")
    problem.plot(figfile=str(args.output_base))
    plt.close("all")

    output_png = Path(f"{args.output_base}-model0.png")
    if not output_png.is_file() or output_png.stat().st_size == 0:
        raise RuntimeError(f"Bumps did not create {output_png}")
    payload = {
        "plotting_only": True,
        "bumps_fitproblem_plot": True,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "data_sha256": sha256_file(args.data),
        "backgrounds_sha256": sha256_file(args.backgrounds),
        "pars_sha256": sha256_file(args.pars),
        "parameter_labels": list(problem.labels()),
        "parameter_values": [float(value) for value in problem.getp()],
        "nllf": nllf,
        "reduced_chi_squared": float(problem.chisq()),
        "model_options": model_options,
        "output_png": str(output_png.resolve()),
        "output_png_sha256": sha256_file(output_png),
    }
    Path(f"{args.output_base}-plot.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
