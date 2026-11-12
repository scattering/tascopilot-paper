#!/usr/bin/env python3
"""Fail-closed runtime/model preflight for the official-Python EPS pilot."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PR110_COMMIT = "1018491992228282d176733a8bb0ad36bad60087"
EXPECTED_HEAD = "82c14f35579d309c1a1a816940920a25d3254897"
EXPECTED_EXTENSION_SHA256 = "f38308200d4e206b06668b02c5f7319dec8fea0b70727229693fcaedcb0e5d6e"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyspinw-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--seed", choices=("all102", "center005"), required=True)
    parser.add_argument(
        "--normalization", choices=("per_scan", "shared_zone_config"), required=True
    )
    parser.add_argument("--domain", choices=("single_y", "equal3"), default="single_y")
    parser.add_argument("--resolution", choices=("legacy", "header_horizontal"), default="legacy")
    parser.add_argument("--qmc", type=int, default=512)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--expected-extension-sha256", default=EXPECTED_EXTENSION_SHA256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.pyspinw_root.resolve()
    if _git(root, "rev-parse", "HEAD") != args.expected_head:
        raise RuntimeError("pySpinW source HEAD is not the pinned official Stampede build")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("pySpinW source checkout is dirty")
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", PR110_COMMIT, "HEAD"],
        check=True,
    )

    import pyspinw
    import pyspinw.rust
    from pyspinw.hamiltonian import Hamiltonian

    pyspinw_file = Path(pyspinw.__file__).resolve()
    rust_file = Path(pyspinw.rust.__file__).resolve()
    runtime_root = Path(sys.prefix).resolve()
    for label, imported_path in (("pyspinw", pyspinw_file), ("pyspinw.rust", rust_file)):
        try:
            imported_path.relative_to(runtime_root)
        except ValueError as exc:
            raise RuntimeError(
                f"{label} import escaped the isolated runtime {runtime_root}: {imported_path}"
            ) from exc
    if _sha256(rust_file) != args.expected_extension_sha256:
        raise RuntimeError("imported pySpinW extension hash differs from the pinned wheel")
    # The extension is not called for intensity; its hash proves this is the
    # validated isolated wheel rather than the old editable checkout.
    signature = inspect.signature(Hamiltonian._energies_and_intensities)
    required = {"q_vectors", "use_rust", "use_rotating", "intensity_unit", "components"}
    if not required <= set(signature.parameters):
        raise RuntimeError(f"official array adapter signature changed: {signature}")
    installed_hamiltonian = Path(inspect.getsourcefile(Hamiltonian) or "").resolve()
    checkout_hamiltonian = root / "pyspinw" / "hamiltonian.py"
    if not checkout_hamiltonian.is_file() or _sha256(installed_hamiltonian) != _sha256(checkout_hamiltonian):
        raise RuntimeError("installed Hamiltonian source differs from the pinned official checkout")

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from srmn2sb2_official_python_intensity_core import (
        OfficialPythonIntensityObjective,
        SEED_PARAMETERS,
        prepare_data,
        python_sperp,
    )

    # Small behavioural sentinel: any accidental Rust path raises immediately.
    class SentinelModel:
        def _energies_and_intensities(self, *, q_vectors, use_rust, **kwargs):
            if use_rust is not False:
                raise AssertionError("Rust intensity path was requested")
            q_vectors = np.asarray(q_vectors)
            energy = np.column_stack([-np.ones(len(q_vectors)), np.ones(len(q_vectors))])
            return energy, np.full_like(energy, 0.5)

    sentinel_energy, sentinel_intensity = python_sperp(
        SentinelModel(), np.asarray([[0.37, 0.21, 1.23], [0.11, 0.29, 2.41]])
    )
    if sentinel_energy.shape != (2, 2) or sentinel_intensity.shape != (2, 2):
        raise RuntimeError("Python-backend call contract smoke failed")

    prepared = prepare_data(
        args.data, args.backgrounds, args.qmc, resolution_policy=args.resolution
    )
    objective = OfficialPythonIntensityObjective(prepared, args.normalization, args.domain)
    sj1, ratio, sjc = SEED_PARAMETERS[args.seed]
    scales = {key: objective.scale_start(key) for key in objective.scale_keys}
    residual = objective.residuals(sj1, ratio, sjc, 0.08853, scales)
    nllf = 0.5*float(np.dot(residual, residual))
    if residual.shape != (prepared.npoints,) or not np.all(np.isfinite(residual)) or not math.isfinite(nllf):
        raise RuntimeError("full objective preflight is non-finite or has the wrong residual length")

    report = objective.provenance()
    report.update({
        "status": "passed",
        "python_executable": sys.executable,
        "pyspinw_source_root": str(root),
        "pyspinw_source_head": args.expected_head,
        "contains_pr110": True,
        "pyspinw_file": str(pyspinw_file),
        "isolated_runtime_root": str(runtime_root),
        "installed_hamiltonian_file": str(installed_hamiltonian),
        "installed_hamiltonian_sha256": _sha256(installed_hamiltonian),
        "pyspinw_rust_file": str(rust_file),
        "pyspinw_rust_sha256": _sha256(rust_file),
        "array_adapter_signature": str(signature),
        "seed": args.seed,
        "seed_parameters": {"SJ1_meV": sj1, "J2_over_J1": ratio, "SJc_meV": sjc},
        "normalization_policy": args.normalization,
        "domain_policy": args.domain,
        "resolution_policy": args.resolution,
        "residual_count": len(residual),
        "initial_nllf": nllf,
        "rust_intensity_sentinel": "passed; production call supplied use_rust=False",
        "outer_bumps_parallelism_allowed": False,
        "required_bumps_cli": "omit --parallel; stock --init=eps, DREAM, then stock serial LM",
    })
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
