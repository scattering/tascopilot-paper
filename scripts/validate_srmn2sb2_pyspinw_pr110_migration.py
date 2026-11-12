#!/usr/bin/env python3
"""Validate the SrMn2Sb2 migration to post-PR110 pySpinW.

This is a behavioural gate, not a fit.  It keeps the January 2026 raw-wrapper
energies as an energy-only oracle, validates corrected S_perp semantics, and
requires Rust and Python to preserve the same energy--intensity eigenmode
pairing.  A build with the known Hermitian-solver pairing defect is reported as
blocked and exits nonzero; its energy-only checks remain separately valid.

The script imports pySpinW from an explicit checkout.  A suitable local
invocation is::

    python \
      scripts/validate_srmn2sb2_pyspinw_pr110_migration.py \
      --pyspinw-root /path/to/pySpinW

It writes only to a temporary directory and stdout unless ``--json-out`` is
explicitly requested.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


PR110_COMMIT = "1018491992228282d176733a8bb0ad36bad60087"
DEFAULT_PYSPINW_ROOT = Path(
    os.environ.get("PYSPINW_ROOT", str(Path(__file__).resolve().parents[2] / "pySpinW"))
)

# The old raw-wrapper run was evaluated at pySpinW commit 260fabb.  Only its
# energies are valid migration oracles; its Sab/S_perp implementation predates
# PR110 and is intentionally absent here.
BRIDGE_Q_RLU = np.array(
    [[0.0, 0.0, 2.0], [0.04, 0.04, 2.0], [0.1, 0.1, 2.0], [0.0, 0.0, 2.1]],
    dtype=float,
)
OLD_RAW_POSITIVE_ENERGIES_MEV = (
    np.array([3.890758280850938]),
    np.array([2.8896678847647843, 4.813639137528557]),
    np.array([8.59884327258573, 9.335454200719049]),
    np.array([7.448803678810655, 8.393871575501334]),
)

# Position-only McNally Eq. (2) known answers for the exchange-only model with
# (SJ1, J2/J1, SJc) = (7.9 meV, 0.165, 0.51 meV).
ANALYTIC_Q_RLU = np.array(
    [[0.1, 0.1, 2.0], [1.0 / 3.0, 1.0 / 3.0, 2.0], [0.0, 0.0, 2.5]],
    dtype=float,
)
ANALYTIC_ACOUSTIC_MEV = np.array([8.59159815, 26.99700000, 24.08684290])

# Generic, nondegenerate points whose SrMn2Sb2 S_perp weights are not invariant
# under reversal of the complete mode axis.  This makes the pairing regression
# sensitive to the upstream Rust Hermitian-solver defect while avoiding a
# Goldstone point or a symmetry-protected degeneracy.
PAIRING_Q_RLU = np.array(
    [[0.37, 0.21, 1.23], [0.11, 0.29, 2.41]],
    dtype=float,
)
PAIRING_ENERGY_ATOL_MEV = 2.0e-8
PAIRING_INTENSITY_ATOL = 2.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_runtime(
    root: Path,
    allow_dirty: bool,
    import_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    import_root = root if import_root is None else import_root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a pySpinW git checkout: {root}")

    checkout_root = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if checkout_root != root:
        raise RuntimeError(f"Expected checkout root {root}, git reports {checkout_root}")

    status = _git(root, "status", "--porcelain", "--untracked-files=all").stdout
    if status and not allow_dirty:
        raise RuntimeError(
            "Refusing a dirty pySpinW checkout; use --allow-dirty only for exploratory runs:\n"
            + status
        )

    ancestor = _git(root, "merge-base", "--is-ancestor", PR110_COMMIT, "HEAD", check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(f"pySpinW HEAD does not contain PR110 ({PR110_COMMIT})")

    # Avoid writing matplotlib/font caches into either repository.
    cache_root = Path(tempfile.gettempdir()) / "srmn2sb2_pyspinw_validation_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if "pyspinw" in sys.modules:
        raise RuntimeError("pyspinw was imported before the explicit runtime boundary")
    sys.path.insert(0, str(import_root))

    import pyspinw
    from pyspinw import (
        Hamiltonian,
        HeisenbergExchange,
        UnitCell,
        axis_anisotropies,
        generate_structure,
    )
    from pyspinw.basis import site_rotations
    import pyspinw.calculations.spinwave as python_spinwave
    import pyspinw.rust as rust

    # Deterministic, local Python fallback checks; the Rust path remains native.
    python_spinwave.get_Executor = lambda: ThreadPoolExecutor
    # Current upstream partitions even a two-Q validation fixture into
    # floor(cpu_count/4) chunks.  On high-core nodes that creates many empty
    # arrays and makes np.concat fail.  One four-core-equivalent worker keeps
    # this read-only fixture nonempty without changing its numerical kernel.
    python_spinwave.multiprocessing.cpu_count = lambda: 4

    pyspinw_file = Path(pyspinw.__file__).resolve()
    rust_file = Path(rust.__file__).resolve()
    if not _inside(pyspinw_file, import_root) or not _inside(rust_file, import_root):
        raise RuntimeError(
            "Import escaped the requested runtime root: "
            f"pyspinw={pyspinw_file}, rust={rust_file}, import_root={import_root}"
        )

    signature = str(inspect.signature(rust.spinwave_calculation))
    required_parameters = ("rlu_to_cart", "positions", "rotating_frame")
    if not all(name in signature for name in required_parameters):
        raise RuntimeError(f"Rust API is not the post-PR110 signature: {signature}")

    rust_source = (root / "src" / "spinwave.rs").read_text()
    source_markers = {
        "site_phase_has_2pi": "J2PI * (q.transpose() * r_i)"
        in rust_source,
        "high_level_uses_fractional_positions": "positions.append(site.ijk)"
        in (root / "pyspinw" / "hamiltonian.py").read_text(),
        "high_level_doubles_anisotropy": "anisotropy_matrix.T, **rust_kw) * 2."
        in (root / "pyspinw" / "hamiltonian.py").read_text(),
    }
    if not all(source_markers.values()):
        raise RuntimeError(f"Required post-PR110 source markers missing: {source_markers}")

    target_candidates = [
        root / "target" / "release" / name
        for name in ("librust.dylib", "librust.so", "rust.dll")
    ]
    target_library = next((path for path in target_candidates if path.exists()), None)
    target_sha = _sha256(target_library) if target_library is not None else None
    rust_sha = _sha256(rust_file)
    if target_sha is not None and target_sha != rust_sha:
        raise RuntimeError(
            f"Imported extension differs from {target_library}; build provenance is ambiguous"
        )

    runtime = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "pyspinw_root": str(root),
        "pyspinw_import_root": str(import_root),
        "pyspinw_file": str(pyspinw_file),
        "git_head": _git(root, "rev-parse", "HEAD").stdout.strip(),
        "git_status": status,
        "contains_pr110": True,
        "rust_extension": str(rust_file),
        "rust_extension_sha256": rust_sha,
        "target_release_library": str(target_library) if target_library is not None else None,
        "target_release_sha256": target_sha,
        "rust_signature": signature,
        "source_markers": source_markers,
        "source_audit": {
            "hermitian_solver_reverses_eigenvector_columns": ".U().reverse_cols()"
            in rust_source,
            "note": (
                "Source text is provenance only; the paired behavioural regression "
                "below is authoritative."
            ),
        },
        "python_fallback_validation_runtime": {
            "executor": "ThreadPoolExecutor",
            "reported_cpu_count": 4,
            "reason": "avoid upstream empty-Q-chunk failure on high-core hosts",
        },
    }
    api = {
        "Hamiltonian": Hamiltonian,
        "HeisenbergExchange": HeisenbergExchange,
        "UnitCell": UnitCell,
        "axis_anisotropies": axis_anisotropies,
        "generate_structure": generate_structure,
        "site_rotations": site_rotations,
        "rust": rust,
        "root": root,
    }
    return runtime, api


def build_srmn2sb2_post_pr110_model(
    api: dict[str, Any],
    *,
    sj1: float = 7.9,
    j2_ratio: float = 0.165,
    sjc: float = 0.51,
    spin: float = 1.5,
    d_old_raw: float = 0.1,
    exchange_factor: float = 2.0,
):
    """Build the post-PR110 representative-bond SrMn2Sb2 model.

    ``exchange_factor=2`` and ``a=d_old_raw/2`` are the calibrated migration
    conversions.  They are explicit arguments so the validator can prove that
    omitting the exchange factor produces half the acoustic energy.
    """

    cell = api["UnitCell"](4.58233, 4.58233, 7.71463, gamma=120.0)
    positions = [[1.0 / 3.0, 2.0 / 3.0, 0.62123], [2.0 / 3.0, 1.0 / 3.0, 1.0 - 0.62123]]
    structure = api["generate_structure"](
        cell,
        positions=positions,
        positions_unit="lu",
        spins=[[0.0, spin, 0.0], [0.0, -spin, 0.0]],
        spins_unit="xyz",
        magnitudes=[spin, spin],
        names=["Mn1", "Mn2"],
    )
    mn1, mn2 = structure.sites

    exchanges = []
    for offset in ((0, 0, 0), (0, 1, 0), (-1, 0, 0)):
        exchanges.append(
            api["HeisenbergExchange"](
                mn1,
                mn2,
                j=exchange_factor * sj1 / spin,
                cell_offset=offset,
                name="J1",
            )
        )
    for site in (mn1, mn2):
        for offset in ((0, 1, 0), (1, 0, 0), (1, 1, 0)):
            exchanges.append(
                api["HeisenbergExchange"](
                    site,
                    site,
                    j=exchange_factor * sj1 * j2_ratio / spin,
                    cell_offset=offset,
                    name="J2",
                )
            )
    for offset in ((0, 0, 1), (-1, 0, 1), (0, 1, 1)):
        exchanges.append(
            api["HeisenbergExchange"](
                mn1,
                mn2,
                j=exchange_factor * sjc / spin,
                cell_offset=offset,
                name="Jc",
            )
        )

    # The high-level adapter multiplies every onsite matrix by two before the
    # kernel call.  Thus API a=D_old/2 reproduces raw-wrapper D_old.
    anisotropies = api["axis_anisotropies"](
        structure,
        a=d_old_raw / 2.0,
        axis=[0.0, 0.0, 1.0],
    )
    return api["Hamiltonian"](structure, exchanges, anisotropies)


def _high_level_q_input(model: Any, q_rlu: Any) -> Any:
    """Bridge post-PR110 high-level APIs that accept arrays or ``Path``."""

    parameters = inspect.signature(model.energies_and_intensities).parameters
    first_parameter = next(iter(parameters), None)
    if first_parameter != "path":
        return np.asarray(q_rlu, dtype=float)
    from pyspinw import Path as SpinWPath

    return SpinWPath(
        np.asarray(q_rlu, dtype=float),
        n_points_per_segment=2,
        avoid_endpoints=False,
    )


def _positive_modes(row: Any, floor: float = 1.0e-4) -> np.ndarray:
    values = np.sort(np.asarray(row).real)
    return values[values > floor]


def _to_raw_inputs(model: Any, api: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror the current high-level adapter and expose the raw-kernel inputs."""

    expanded = model.expanded()
    sites = expanded.structure.sites
    spins = np.asarray([site.base_spin for site in sites], dtype=float)
    positions = np.asarray([site.ijk for site in sites], dtype=float)
    rotations = np.asarray(api["site_rotations"](spins), dtype=complex, order="F")
    magnitudes = np.linalg.norm(spins, axis=1)
    uid_to_index = {site._unique_id: index for index, site in enumerate(sites)}

    couplings = []
    for exchange in expanded.exchanges:
        i = uid_to_index[exchange.site_1._unique_id]
        j = uid_to_index[exchange.site_2._unique_id]
        matrix = np.asarray(exchange.exchange_matrix, dtype=complex, order="F")
        offset = np.asarray(exchange.cell_offset.vector, dtype=float)
        couplings.append(api["rust"].Coupling(i, j, matrix, offset))
        couplings.append(api["rust"].Coupling(j, i, matrix.T, -offset))

    exchange_coupling_count = len(couplings)
    for anisotropy in expanded.anisotropies:
        index = uid_to_index[anisotropy.site._unique_id]
        matrix = np.asarray(
            anisotropy.anisotropy_matrix.T * 2.0,
            dtype=complex,
            order="F",
        )
        couplings.append(
            api["rust"].Coupling(index, index, matrix, np.zeros(3, dtype=float))
        )

    rlu_to_cart = np.linalg.inv(expanded.structure.unit_cell._xyz).T * 2.0 * np.pi
    raw = {
        "rotations": rotations,
        "magnitudes": magnitudes,
        "couplings": couplings,
        "positions": positions,
        "rlu_to_cart": rlu_to_cart,
        "field": None,
        "rotating_frame": None,
    }
    metadata = {
        "site_count": len(sites),
        "representative_exchange_count": len(expanded.exchanges),
        "raw_directed_exchange_count": exchange_coupling_count,
        "raw_anisotropy_count": len(expanded.anisotropies),
        "raw_total_coupling_count": len(couplings),
        "fractional_positions": positions.tolist(),
        "rlu_to_cart": rlu_to_cart.tolist(),
    }
    return raw, metadata


def _raw_call(
    api: dict[str, Any],
    raw: dict[str, Any],
    q_rlu: np.ndarray,
    *,
    save_sab: bool,
    positions: np.ndarray | None = None,
):
    rust = api["rust"]
    has_legacy_sab_entrypoint = hasattr(rust, "spinwave_calculation_Sab")
    function = (
        rust.spinwave_calculation_Sab
        if save_sab and has_legacy_sab_entrypoint
        else rust.spinwave_calculation
    )
    keyword_arguments = dict(
        rotations=raw["rotations"],
        magnitudes=raw["magnitudes"],
        q_vectors=np.asarray(q_rlu, dtype=float),
        couplings=raw["couplings"],
        positions=raw["positions"] if positions is None else positions,
        rlu_to_cart=raw["rlu_to_cart"],
        field=None,
        rotating_frame=None,
    )
    signature = inspect.signature(function).parameters
    if "save_sab" in signature:
        keyword_arguments["save_sab"] = save_sab
    if "save_wavefunctions" in signature:
        keyword_arguments["save_wavefunctions"] = False
    result = function(**keyword_arguments)
    if save_sab and not has_legacy_sab_entrypoint:
        return result[:3]
    return result


def _project_sab(sab: np.ndarray, q_vectors: np.ndarray, rlu_to_cart: np.ndarray) -> np.ndarray:
    q_cart = np.asarray(q_vectors) @ np.asarray(rlu_to_cart)
    q_hat = q_cart / np.linalg.norm(q_cart, axis=1)[:, None]
    projector = np.eye(3)[None, :, :] - np.einsum("qa,qb->qab", q_hat, q_hat)
    return np.einsum("qmab,qab->qm", np.asarray(sab), projector).real


def _validate_structure(model: Any) -> dict[str, Any]:
    counts = Counter(exchange.name for exchange in model.exchanges)
    expected = Counter({"J1": 3, "J2": 6, "Jc": 3})
    if counts != expected:
        raise AssertionError(f"Representative-bond counts differ: {counts} != {expected}")

    bonds = Counter(
        (
            exchange.name,
            exchange.site_1.name,
            exchange.site_2.name,
            exchange.cell_offset.as_tuple,
        )
        for exchange in model.exchanges
    )
    expected_bonds = Counter(
        [("J1", "Mn1", "Mn2", offset) for offset in ((0, 0, 0), (0, 1, 0), (-1, 0, 0))]
        + [
            ("J2", site, site, offset)
            for site in ("Mn1", "Mn2")
            for offset in ((0, 1, 0), (1, 0, 0), (1, 1, 0))
        ]
        + [("Jc", "Mn1", "Mn2", offset) for offset in ((0, 0, 1), (-1, 0, 1), (0, 1, 1))]
    )
    if bonds != expected_bonds:
        raise AssertionError(f"Representative-bond topology differs: {bonds} != {expected_bonds}")

    anisotropy_matrices = np.asarray(
        [anisotropy.anisotropy_matrix for anisotropy in model.anisotropies]
    )
    expected_anisotropy = np.diag([0.0, 0.0, 0.05])
    np.testing.assert_allclose(
        anisotropy_matrices,
        np.repeat(expected_anisotropy[None, :, :], 2, axis=0),
        atol=1.0e-14,
        rtol=0.0,
    )
    return {
        "representative_bonds": dict(sorted(counts.items())),
        "representative_bond_records": [
            {
                "name": name,
                "site_1": site_1,
                "site_2": site_2,
                "cell_offset": list(offset),
            }
            for name, site_1, site_2, offset in sorted(bonds)
        ],
        "anisotropy_api_matrix_per_site": expected_anisotropy.tolist(),
        "anisotropy_kernel_matrix_per_site": (2.0 * expected_anisotropy).tolist(),
    }


def _validate_energy_bridge(model: Any) -> dict[str, Any]:
    energies, _ = model.energies_and_intensities(
        _high_level_q_input(model, BRIDGE_Q_RLU),
        use_rust=True,
        use_rotating=False,
        intensity_unit="spin",
    )
    observed = [_positive_modes(row) for row in np.asarray(energies)]
    maximum_error = 0.0
    for q, actual, expected in zip(
        BRIDGE_Q_RLU, observed, OLD_RAW_POSITIVE_ENERGIES_MEV, strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2.0e-8, rtol=1.0e-10)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    return {
        "q_rlu": BRIDGE_Q_RLU.tolist(),
        "old_260fabb_positive_mev": [values.tolist() for values in OLD_RAW_POSITIVE_ENERGIES_MEV],
        "post_pr110_positive_mev": [values.tolist() for values in observed],
        "max_abs_error_mev": maximum_error,
        "old_intensities_used": False,
    }


def _validate_paper_convention(api: dict[str, Any]) -> dict[str, Any]:
    full = build_srmn2sb2_post_pr110_model(api, d_old_raw=0.0, exchange_factor=2.0)
    half = build_srmn2sb2_post_pr110_model(api, d_old_raw=0.0, exchange_factor=1.0)
    full_energies = np.asarray(
        full.energies_and_intensities(
            _high_level_q_input(full, ANALYTIC_Q_RLU),
            use_rust=True,
            use_rotating=False,
            intensity_unit="spin",
        )[0]
    )
    half_energies = np.asarray(
        half.energies_and_intensities(
            _high_level_q_input(half, ANALYTIC_Q_RLU),
            use_rust=True,
            use_rotating=False,
            intensity_unit="spin",
        )[0]
    )
    full_acoustic = np.array([_positive_modes(row)[0] for row in full_energies])
    half_acoustic = np.array([_positive_modes(row)[0] for row in half_energies])
    np.testing.assert_allclose(full_acoustic, ANALYTIC_ACOUSTIC_MEV, atol=8.0e-8, rtol=1.0e-9)
    np.testing.assert_allclose(half_acoustic, full_acoustic / 2.0, atol=8.0e-8, rtol=1.0e-9)

    other_spin = build_srmn2sb2_post_pr110_model(
        api,
        spin=2.5,
        d_old_raw=0.0,
        exchange_factor=2.0,
    )
    other_energies = np.asarray(
        other_spin.energies_and_intensities(
            _high_level_q_input(other_spin, ANALYTIC_Q_RLU),
            use_rust=True,
            use_rotating=False,
            intensity_unit="spin",
        )[0]
    )
    other_acoustic = np.array([_positive_modes(row)[0] for row in other_energies])
    np.testing.assert_allclose(other_acoustic, full_acoustic, atol=8.0e-8, rtol=1.0e-9)

    return {
        "q_rlu": ANALYTIC_Q_RLU.tolist(),
        "analytic_acoustic_mev": ANALYTIC_ACOUSTIC_MEV.tolist(),
        "post_pr110_j_equals_2sj_over_s_mev": full_acoustic.tolist(),
        "post_pr110_j_equals_sj_over_s_mev": half_acoustic.tolist(),
        "half_energy_ratio": (half_acoustic / full_acoustic).tolist(),
        "spin_invariance_s_1p5_vs_2p5_max_abs_mev": float(
            np.max(np.abs(other_acoustic - full_acoustic))
        ),
    }


def _validate_sperp(model: Any, api: dict[str, Any]) -> dict[str, Any]:
    raw, raw_metadata = _to_raw_inputs(model, api)
    q = np.array([[0.37, 0.21, 1.23], [0.11, 0.29, 2.41]], dtype=float)
    raw_energy, raw_intensity, sab = _raw_call(api, raw, q, save_sab=True)
    raw_energy = np.asarray(raw_energy)
    raw_intensity = np.asarray(raw_intensity)
    sab = np.asarray(sab)

    projected = _project_sab(sab, q, raw["rlu_to_cart"])
    np.testing.assert_allclose(projected, raw_intensity, atol=2.0e-10, rtol=2.0e-12)

    # A deliberate wrong-units projection must not pass.  This catches an
    # accidental use of HKL components as Cartesian directions.
    wrong_projected = _project_sab(sab, q, np.eye(3))
    wrong_projection_difference = float(np.max(np.abs(wrong_projected - raw_intensity)))
    if wrong_projection_difference <= 1.0e-4:
        raise AssertionError("The q-unit sensitivity check was not discriminating")

    high_energy, high_intensity = model.energies_and_intensities(
        _high_level_q_input(model, q),
        use_rust=True,
        use_rotating=False,
        intensity_unit="spin",
    )
    np.testing.assert_allclose(np.asarray(high_energy), raw_energy, atol=2.0e-10, rtol=2.0e-12)
    np.testing.assert_allclose(
        np.asarray(high_intensity), raw_intensity, atol=2.0e-10, rtol=2.0e-12
    )

    _, cell_intensity = model.energies_and_intensities(
        _high_level_q_input(model, q),
        use_rust=True,
        use_rotating=False,
        intensity_unit="cell",
    )
    np.testing.assert_allclose(
        np.asarray(cell_intensity),
        raw_metadata["site_count"] * raw_intensity,
        atol=4.0e-10,
        rtol=2.0e-12,
    )

    # Corrected site phases are exp(i*2*pi*q.r).  Moving one basis position by
    # delta with q.delta=1 must leave S_perp unchanged; a half-period shift must
    # change it, proving the test is sensitive rather than accidentally flat.
    phase_q = np.array([[0.37, 0.21, 1.23]], dtype=float)
    base_intensity = np.asarray(_raw_call(api, raw, phase_q, save_sab=False)[1])
    delta = phase_q[0] / np.dot(phase_q[0], phase_q[0])
    shifted_positions = np.array(raw["positions"], copy=True)
    shifted_positions[1] += delta
    shifted_intensity = np.asarray(
        _raw_call(
            api,
            raw,
            phase_q,
            save_sab=False,
            positions=shifted_positions,
        )[1]
    )
    phase_period_error = float(np.max(np.abs(shifted_intensity - base_intensity)))
    np.testing.assert_allclose(shifted_intensity, base_intensity, atol=2.0e-10, rtol=2.0e-12)

    half_shifted_positions = np.array(raw["positions"], copy=True)
    half_shifted_positions[1] += 0.5 * delta
    half_shifted_intensity = np.asarray(
        _raw_call(
            api,
            raw,
            phase_q,
            save_sab=False,
            positions=half_shifted_positions,
        )[1]
    )
    half_period_difference = float(
        np.max(np.abs(half_shifted_intensity - base_intensity))
    )
    if half_period_difference <= 1.0e-4:
        raise AssertionError("The 2pi site-phase test was not discriminating")

    return {
        "scope": (
            "Rust internal projector and high-level/raw adapter consistency only; "
            "this does not validate energy--Sab eigenmode pairing"
        ),
        "q_rlu": q.tolist(),
        "raw_adapter_metadata": raw_metadata,
        "projector_identity_max_abs": float(np.max(np.abs(projected - raw_intensity))),
        "wrong_hkl_direction_max_abs": wrong_projection_difference,
        "high_level_raw_energy_max_abs": float(
            np.max(np.abs(np.asarray(high_energy) - raw_energy))
        ),
        "high_level_raw_sperp_max_abs": float(
            np.max(np.abs(np.asarray(high_intensity) - raw_intensity))
        ),
        "per_cell_over_per_spin": raw_metadata["site_count"],
        "integer_phase_period_max_abs": phase_period_error,
        "half_phase_period_max_abs": half_period_difference,
    }


def _canonicalize_paired_modes(
    energies: Any, intensities: Any, *, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    """Order modes by energy while applying that same order to intensity."""

    energy = np.asarray(energies)
    intensity = np.asarray(intensities, dtype=float)
    if energy.ndim != 2 or intensity.shape != energy.shape:
        raise AssertionError(
            f"{backend} mode arrays must have matching (Q, mode) shapes; "
            f"got energy={energy.shape}, intensity={intensity.shape}"
        )
    maximum_imaginary = float(np.max(np.abs(energy.imag)))
    if maximum_imaginary > 1.0e-8:
        raise AssertionError(
            f"{backend} pairing fixture unexpectedly has complex energies; "
            f"max |Im(E)|={maximum_imaginary}"
        )
    real_energy = energy.real
    if not np.all(np.isfinite(real_energy)) or not np.all(np.isfinite(intensity)):
        raise AssertionError(f"{backend} pairing fixture returned non-finite values")
    order = np.argsort(real_energy, axis=1, kind="stable")
    return (
        np.take_along_axis(real_energy, order, axis=1),
        np.take_along_axis(intensity, order, axis=1),
    )


def _validate_paired_rust_python_modes(model: Any) -> dict[str, Any]:
    """Compare Rust/Python S_perp only after preserving each eigenpair mapping."""

    rust_result = model.energies_and_intensities(
        _high_level_q_input(model, PAIRING_Q_RLU),
        use_rust=True,
        use_rotating=False,
        intensity_unit="spin",
    )
    python_result = model.energies_and_intensities(
        _high_level_q_input(model, PAIRING_Q_RLU),
        use_rust=False,
        use_rotating=False,
        intensity_unit="spin",
    )
    rust_energy, rust_intensity = _canonicalize_paired_modes(
        *rust_result, backend="Rust"
    )
    python_energy, python_intensity = _canonicalize_paired_modes(
        *python_result, backend="Python"
    )
    if rust_energy.shape != python_energy.shape:
        raise AssertionError(
            "Rust and Python returned different mode-array shapes: "
            f"{rust_energy.shape} != {python_energy.shape}"
        )

    # The fixture must remain nondegenerate, otherwise an arbitrary unitary
    # rotation inside a degenerate subspace would make columnwise comparison
    # ambiguous even for a correct implementation.
    minimum_mode_separation = float(
        min(np.min(np.diff(row)) for row in python_energy)
    )
    if minimum_mode_separation <= 1.0e-5:
        raise AssertionError(
            "Rust/Python pairing fixture is no longer nondegenerate; "
            f"minimum mode separation={minimum_mode_separation} meV"
        )

    energy_max_abs = float(np.max(np.abs(rust_energy - python_energy)))
    paired_intensity_max_abs = float(
        np.max(np.abs(rust_intensity - python_intensity))
    )
    # Diagnostic signature only.  Reversing output intensities is not a valid
    # production repair because solver paths and future eigenvalue orderings can
    # differ; the upstream solver must preserve one shared eigenpair permutation.
    reversed_axis_max_abs = float(
        np.max(np.abs(rust_intensity - python_intensity[:, ::-1]))
    )
    energies_agree = bool(energy_max_abs <= PAIRING_ENERGY_ATOL_MEV)
    paired_intensities_agree = bool(
        np.allclose(
            rust_intensity,
            python_intensity,
            atol=PAIRING_INTENSITY_ATOL,
            rtol=2.0e-8,
        )
    )
    passed = energies_agree and paired_intensities_agree
    reversal_signature = bool(
        energies_agree
        and not paired_intensities_agree
        and np.allclose(
            rust_intensity,
            python_intensity[:, ::-1],
            atol=PAIRING_INTENSITY_ATOL,
            rtol=2.0e-8,
        )
    )
    blocker = None
    if not passed:
        blocker = (
            "Rust-backed SrMn2Sb2 Sab/S_perp is not validated because Rust and "
            "Python do not preserve the same energy--intensity eigenmode pairs. "
            "Use use_rust=False for intensity calculations until upstream applies "
            "one shared eigenpair permutation to eigenvalues and eigenvectors. "
            "Do not apply a post-hoc full-axis reversal in production."
        )
    return {
        "passed": passed,
        "q_rlu": PAIRING_Q_RLU.tolist(),
        "observable": "S_perp_per_spin",
        "canonicalization": (
            "Each backend is sorted by energy and the identical backend-specific "
            "permutation is applied to its S_perp; intensities are never sorted alone."
        ),
        "minimum_python_mode_separation_mev": minimum_mode_separation,
        "energy_max_abs_mev": energy_max_abs,
        "paired_sperp_max_abs": paired_intensity_max_abs,
        "full_mode_axis_reversal_signature_max_abs": reversed_axis_max_abs,
        "full_mode_axis_reversal_signature_detected": reversal_signature,
        "rust_energy_only_valid_for_fixture": energies_agree,
        "rust_sab_sperp_valid_for_fixture": paired_intensities_agree,
        "required_action": (
            None
            if passed
            else "Use use_rust=False for intensities; retain Rust only for energy-only work."
        ),
        "blocker": blocker,
    }


def _validate_python_matlab_fixture(api: dict[str, Any], skip: bool) -> dict[str, Any]:
    root = api["root"]
    fixture = root / "tests" / "matlab_test_data.mat"
    test_file = root / "tests" / "test_interface.py"
    if skip:
        return {"skipped": True, "reason": "--skip-matlab-fixture"}
    if not fixture.exists() or not test_file.exists():
        raise RuntimeError("The official upstream MATLAB parity fixture/test is absent")

    before = _git(root, "status", "--porcelain", "--untracked-files=all").stdout
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not old_pythonpath else f"{root}{os.pathsep}{old_pythonpath}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="srmn2sb2_pyspinw_matlab_") as temporary:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--disable-warnings",
                f"{test_file}::test_antiferro_chain",
            ],
            cwd=temporary,
            env=env,
            capture_output=True,
            text=True,
        )
    after = _git(root, "status", "--porcelain", "--untracked-files=all").stdout
    if before != after:
        raise RuntimeError("The read-only MATLAB fixture check changed the pySpinW checkout")
    if result.returncode != 0:
        raise AssertionError(
            "Official Python/MATLAB parity test failed:\n"
            + result.stdout
            + "\n"
            + result.stderr
        )
    return {
        "skipped": False,
        "fixture": str(fixture),
        "fixture_sha256": _sha256(fixture),
        "test": f"{test_file}::test_antiferro_chain",
        "result": result.stdout.strip().splitlines()[-1],
        "engine": "post-PR110 high-level Python backend only (the test hard-codes use_rust=False)",
        "rust_matlab_parity_established": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyspinw-root",
        type=Path,
        default=DEFAULT_PYSPINW_ROOT,
        help="Explicit official pySpinW checkout/build to validate",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty external checkout (not suitable for provenance claims)",
    )
    parser.add_argument(
        "--pyspinw-import-root",
        type=Path,
        help=(
            "Optional directory containing the built pyspinw package. Use this "
            "when validating a wheel installed outside the clean source checkout."
        ),
    )
    parser.add_argument(
        "--skip-matlab-fixture",
        action="store_true",
        help=(
            "Skip the official upstream Python-backend/MATLAB "
            "antiferromagnetic-chain fixture"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional local derived JSON report path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime, api = _load_runtime(
        args.pyspinw_root,
        allow_dirty=args.allow_dirty,
        import_root=args.pyspinw_import_root,
    )
    model = build_srmn2sb2_post_pr110_model(api)
    pairing = _validate_paired_rust_python_modes(model)
    blockers = [pairing["blocker"]] if pairing["blocker"] is not None else []
    report = {
        "status": "blocked" if blockers else "pass",
        "scope": "SrMn2Sb2 260fabb raw-wrapper to post-PR110 high-level migration",
        "blockers": blockers,
        "runtime": runtime,
        "checks": {
            "representative_model": _validate_structure(model),
            "old_to_new_energy_bridge": _validate_energy_bridge(model),
            "paper_product_convention": _validate_paper_convention(api),
            "rust_internal_sperp_consistency": _validate_sperp(model, api),
            "paired_rust_python_eigenmodes": pairing,
            "official_python_matlab_fixture": _validate_python_matlab_fixture(
                api, skip=args.skip_matlab_fixture
            ),
        },
        "boundary": {
            "old_260fabb_energies_are_oracle": True,
            "old_260fabb_intensities_are_oracle": False,
            "rust_energy_only_use_allowed": bool(
                pairing["rust_energy_only_valid_for_fixture"]
            ),
            "rust_sab_sperp_use_allowed": bool(
                pairing["rust_sab_sperp_valid_for_fixture"]
            ),
            "intensity_backend_until_upstream_fix": (
                "use_rust=True"
                if pairing["passed"]
                else "use_rust=False"
            ),
            "posthoc_mode_axis_reversal_allowed": False,
            "form_factor_included": False,
            "resolution_convolution_included": False,
            "fit_parameters_estimated": False,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
