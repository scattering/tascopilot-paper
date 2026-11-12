"""CIF-based alignment dashboard backend.

Backs the workbench `/alignment` page: parse an uploaded CIF with pymatgen,
calculate neutron nuclear structure factors through the configured GSAS-II
runtime (never internally), suggest alignment peaks in a user-chosen
scattering plane with the existing alignment-peak criteria, build the
Brillouin-zone section for that plane, and overlay powder rings for selected
sample-environment materials.

Boundaries preserved:

- F^2 always comes from the GSAS-II runtime adapter; if that runtime is
  unavailable the report fails closed instead of rolling its own
  crystallography.
- Peak suggestions reuse `_build_peak_candidates_from_gsasii_reflections`
  (plane status, out-of-plane acceptance, background warnings), so the
  dashboard and the CLI workflow share one set of criteria.
- Everything here is planning evidence requiring instrument-scientist review,
  not approved instrument operation.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

# Sample-environment overlay materials with public, textbook lattice data.
# These are deliberate checked-in constants (not ICSD content): the asymmetric
# unit is a single atom at the origin and GSAS-II applies Fm-3m symmetry.
ENVIRONMENT_MATERIALS: dict[str, dict[str, Any]] = {
    "aluminum": {
        "material_id": "aluminum",
        "label": "Aluminum",
        "formula": "Al",
        "space_group": "F m -3 m",
        "space_group_number": 225,
        "lattice": {"a": 4.0495, "b": 4.0495, "c": 4.0495, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "atoms": [{"symbol": "Al", "x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0}],
        "source": "textbook fcc lattice constant (4.0495 A at room temperature)",
    },
    "copper": {
        "material_id": "copper",
        "label": "Copper",
        "formula": "Cu",
        "space_group": "F m -3 m",
        "space_group_number": 225,
        "lattice": {"a": 3.6149, "b": 3.6149, "c": 3.6149, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "atoms": [{"symbol": "Cu", "x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0}],
        "source": "textbook fcc lattice constant (3.6149 A at room temperature)",
    },
}


def parse_cif_structure(cif_path: str | Path) -> dict[str, Any]:
    """Parse a CIF file via :func:`parse_cif_text` (local-workbench convenience)."""
    path = Path(cif_path).expanduser()
    if not path.exists():
        return {"status": "missing_input", "missing_inputs": ["cif_path"], "warnings": [f"CIF file does not exist: {path}"]}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": "error", "missing_inputs": ["readable_cif"], "warnings": [f"CIF file could not be read: {exc}"]}
    summary = parse_cif_text(content)
    if summary.get("status") == "success":
        summary["cif_path"] = str(path)
    return summary


def parse_cif_text(content: str) -> dict[str, Any]:
    """Parse CIF text in memory into the structure payload the GSAS-II adapter expects.

    Uses pymatgen's CIF parser and the conventional standard setting so the
    space-group symbol matches what GSAS-II resolves. Nothing is written to
    disk: uploaded structures stay in memory. Fails closed with a typed status
    when pymatgen is unavailable or the CIF does not parse.
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except Exception as exc:  # pragma: no cover - optional dependency path.
        return {"status": "unavailable", "missing_inputs": ["pymatgen"], "warnings": [f"pymatgen is required for CIF parsing: {exc}"]}
    try:
        structure = Structure.from_str(content, fmt="cif")
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
        conventional = analyzer.get_conventional_standard_structure()
        symmetrized = SpacegroupAnalyzer(conventional, symprec=1e-3).get_symmetrized_structure()
    except Exception as exc:
        return {"status": "error", "missing_inputs": ["parseable_cif"], "warnings": [f"CIF parsing failed: {type(exc).__name__}: {exc}"]}
    atoms: list[dict[str, Any]] = []
    for group in symmetrized.equivalent_sites:
        site = group[0]
        for species, occupancy in site.species.items():
            atoms.append(
                {
                    "symbol": species.symbol,
                    "x": float(site.frac_coords[0]),
                    "y": float(site.frac_coords[1]),
                    "z": float(site.frac_coords[2]),
                    "occupancy": float(occupancy),
                }
            )
    lattice = conventional.lattice
    return {
        "status": "success",
        "formula": conventional.composition.reduced_formula,
        "full_formula": str(conventional.composition.formula),
        "space_group": analyzer.get_space_group_symbol(),
        "space_group_number": analyzer.get_space_group_number(),
        "crystal_system": analyzer.get_crystal_system(),
        "lattice": {
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
        },
        "atoms": atoms,
        "asymmetric_site_count": len(atoms),
        "warnings": [
            "Structure summary comes from the conventional standard setting of the uploaded CIF.",
            "CIF content is untrusted user input; results are planning evidence only.",
        ],
    }


def _structure_from_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Resolve the structure from in-memory CIF content (preferred) or a path."""
    content = arguments.get("cif_content")
    if content:
        return parse_cif_text(str(content))
    return parse_cif_structure(str(arguments.get("cif_path") or ""))

def alignment_dashboard_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """Build the full alignment-dashboard payload for one CIF and plane."""
    from tas_copilot_core.structure_factors import (
        _build_peak_candidates_from_gsasii_reflections,
        _run_gsasii_reflections,
        _StructureRecord,
    )

    structure = _structure_from_arguments(arguments)
    if structure.get("status") != "success":
        return {"status": structure.get("status", "error"), "structure": structure, "warnings": structure.get("warnings", [])}

    orient1 = [float(value) for value in (arguments.get("orient1") or [1, 0, 0])]
    orient2 = [float(value) for value in (arguments.get("orient2") or [0, 1, 0])]
    energy_mev = float(arguments.get("energy_mev") or 14.7)
    a4_max_deg = float(arguments.get("a4_max_deg") or 120.0)
    top_k = int(arguments.get("top_k") or 15)
    max_listed = int(arguments.get("max_listed_reflections") or 200)
    requested_materials = [str(item).lower() for item in (arguments.get("environment_materials") or [])]
    warnings: list[str] = list(structure.get("warnings") or [])

    lattice = structure["lattice"]
    record = _StructureRecord(
        idnum=-1,
        coll_code=None,
        year=None,
        sgr=structure["space_group"],
        sgr_num=structure["space_group_number"],
        lattice=(lattice["a"], lattice["b"], lattice["c"], lattice["alpha"], lattice["beta"], lattice["gamma"]),
        r_value=None,
        sum_form=structure["formula"],
        struct_form=structure.get("full_formula"),
        chem_name=None,
        atoms=structure["atoms"],
        symops=[],
    )
    runtime_result = _run_gsasii_reflections(arguments, record, energy_mev=energy_mev, a4_max_deg=a4_max_deg)
    adapter_output = runtime_result.output or {}
    if runtime_result.status.value != "success" or adapter_output.get("status") != "success":
        warnings.extend(error.message for error in runtime_result.errors)
        warnings.extend(str(error) for error in adapter_output.get("errors") or [])
        return {
            "status": "unavailable",
            "structure": structure,
            "missing_inputs": ["gsasii_structure_factor_runtime"],
            "warnings": warnings + ["Structure factors require the configured GSAS-II runtime; the dashboard fails closed without it."],
        }
    reflections = list(adapter_output.get("reflections") or [])
    warnings.extend(adapter_output.get("warnings") or [])
    unique_reflection_count = len(reflections)
    # GSAS-II returns one representative per symmetry star; single-crystal
    # alignment needs the full star (e.g. (030) alongside (300)), and an
    # in-plane equivalent of an out-of-plane representative must be eligible
    # for suggestion.
    reflections, expansion_warnings = _expand_reflections_by_symmetry(arguments, structure, reflections, orient1, orient2)
    warnings.extend(expansion_warnings)

    scattering_plane = _plane_string(orient1, orient2)
    background_materials = [
        {
            "material_id": ENVIRONMENT_MATERIALS[name]["material_id"],
            "label": ENVIRONMENT_MATERIALS[name]["label"],
            "formula": ENVIRONMENT_MATERIALS[name]["formula"],
            "enabled": True,
            "source": "alignment_dashboard_checkbox",
        }
        for name in requested_materials
        if name in ENVIRONMENT_MATERIALS
    ]
    peaks, peak_warnings = _build_peak_candidates_from_gsasii_reflections(
        reflections,
        record,
        scattering_plane=scattering_plane,
        top_k=top_k,
        background_materials=background_materials,
        background_peak_tables=[],
        background_overlap_tolerance_deg=float(arguments.get("background_overlap_tolerance_deg") or 0.5),
        low_q_tilt_deg=float(arguments.get("out_of_plane_low_q_deg") or 2.0),
        high_q_tilt_deg=float(arguments.get("out_of_plane_high_q_deg") or 1.0),
        max_out_of_plane_deg=float(arguments.get("out_of_plane_max_deg") or 5.0),
        sample_thickness_mm=None,
    )
    warnings.extend(peak_warnings)
    suggested = [peak.model_dump(mode="json") for peak in peaks]

    bz_section, basis, bz_warnings = _bz_section(lattice, orient1, orient2)
    warnings.extend(bz_warnings)
    peak_points = _peak_plane_points(suggested, lattice, basis) if basis else []
    plane_lattice = _plane_lattice_summary(lattice, basis) if basis else None
    if plane_lattice and plane_lattice.get("warnings"):
        warnings.extend(plane_lattice.pop("warnings"))

    environment_rings = _environment_rings(arguments, requested_materials, energy_mev=energy_mev, a4_max_deg=a4_max_deg)
    for ring_set in environment_rings:
        warnings.extend(ring_set.pop("warnings", []))

    report = {
        "status": "success",
        "structure": structure,
        "scattering_plane": scattering_plane,
        "energy_mev": energy_mev,
        "a4_max_deg": a4_max_deg,
        "gsasii_backend": adapter_output.get("backend"),
        "reflection_count": len(reflections),
        "unique_star_count": unique_reflection_count,
        "reflections": reflections[:max_listed],
        "suggested_peaks": suggested,
        "bz_section": bz_section,
        "plane_basis": basis,
        "plane_lattice": plane_lattice,
        "peak_points": peak_points,
        "environment_rings": environment_rings,
        "warnings": warnings,
        "safety_note": "Planning evidence requiring instrument-scientist review; not an approved instrument operation.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_dir = arguments.get("output_dir")
    if output_dir:
        out = Path(str(output_dir)).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        import json as _json

        (out / "alignment_dashboard_report.json").write_text(_json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["artifacts"] = {"json": str(out / "alignment_dashboard_report.json")}
    return report


def vector_tools_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """Plane vector utilities: angle between HKLs, in-plane orthogonal, equivalents.

    The angle and orthogonal come from the icp-lattice-calculator metric
    through the registry (`tas.ubmatrix.calculate_reciprocal_angle` /
    `calculate_inplane_orthogonal`); symmetry-equivalent reflections come from
    GSAS-II GenHKLf through the runtime adapter. Each section fails soft so a
    missing plugin or runtime degrades to a status note.
    """
    from tas_copilot_contracts import ToolCall
    from tas_copilot_core.workflows.common import default_runner

    structure = _structure_from_arguments(arguments)
    if structure.get("status") != "success":
        return {"status": structure.get("status", "error"), "structure": structure, "warnings": structure.get("warnings", [])}
    orient1 = [float(value) for value in (arguments.get("orient1") or [1, 0, 0])]
    orient2 = [float(value) for value in (arguments.get("orient2") or [0, 1, 0])]
    v1 = [float(value) for value in (arguments.get("v1") or [])]
    v2 = [float(value) for value in (arguments.get("v2") or [])]
    if not v1:
        return {"status": "missing_input", "missing_inputs": ["v1"], "warnings": ["Provide at least one HKL vector."]}
    lattice = dict(structure["lattice"])
    lattice["orient1"] = [orient1]
    lattice["orient2"] = [orient2]
    runner = default_runner()
    report: dict[str, Any] = {
        "status": "success",
        "formula": structure.get("formula"),
        "space_group": structure.get("space_group"),
        "scattering_plane": _plane_string(orient1, orient2),
        "v1": v1,
        "warnings": [],
    }

    if v2:
        angle_result = runner.run(
            ToolCall(
                tool_namespace="tas.ubmatrix",
                tool_name="calculate_reciprocal_angle",
                arguments={"lattice": lattice, "v1": v1, "v2": v2},
                caller_workflow="alignment-dashboard-vector-tools",
            )
        )
        report["angle"] = (
            angle_result.output
            if angle_result.status.value == "success"
            else {"status": angle_result.status.value, "note": "tas.ubmatrix.calculate_reciprocal_angle unavailable; install the rescalculator plugin."}
        )

    orthogonal_result = runner.run(
        ToolCall(
            tool_namespace="tas.ubmatrix",
            tool_name="calculate_inplane_orthogonal",
            arguments={"lattice": lattice, "v": v1},
            caller_workflow="alignment-dashboard-vector-tools",
        )
    )
    report["inplane_orthogonal"] = (
        orthogonal_result.output
        if orthogonal_result.status.value == "success"
        else {"status": orthogonal_result.status.value, "note": "tas.ubmatrix.calculate_inplane_orthogonal unavailable; install the rescalculator plugin."}
    )

    equivalents_targets: list[list[int]] = []
    if all(abs(value - round(value)) < 1e-6 for value in v1):
        equivalents_targets.append([int(round(value)) for value in v1])
    rational = (report.get("inplane_orthogonal") or {}).get("orthogonal_hkl_rational")
    if rational:
        equivalents_targets.append([int(value) for value in rational])
    report["equivalents"] = _equivalent_reflections(arguments, structure, equivalents_targets, orient1, orient2) if equivalents_targets else {
        "status": "skipped_non_integer_hkl",
        "note": "Symmetry equivalents need integer HKL input.",
    }
    return report


def _expand_reflections_by_symmetry(
    arguments: dict[str, Any],
    structure: dict[str, Any],
    reflections: list[dict[str, Any]],
    orient1: list[float],
    orient2: list[float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand star representatives into all symmetry-equivalent reflections.

    Neutron nuclear |F|^2, d, |Q|, and two-theta are invariant across the star
    (including Friedel mates), so each equivalent inherits the representative's
    values. Fails soft: if the equivalents runtime is unavailable, the
    unexpanded list is returned with a warning.
    """
    if not reflections:
        return reflections, []
    targets = [[int(item["h"]), int(item["k"]), int(item["l"])] for item in reflections]
    equivalents = _equivalent_reflections(arguments, structure, targets, orient1, orient2)
    if equivalents.get("status") != "success":
        return reflections, ["Symmetry-star expansion unavailable; the reflection list shows one representative per star."]
    star_by_hkl = {tuple(entry["hkl"]): entry for entry in equivalents.get("results") or []}
    expanded: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for reflection in reflections:
        representative = (int(reflection["h"]), int(reflection["k"]), int(reflection["l"]))
        star = star_by_hkl.get(representative)
        mates = star.get("equivalents") if star else None
        if not mates:
            mates = [{"hkl": list(representative), "friedel_mate": False}]
        for mate in mates:
            hkl = tuple(int(value) for value in mate["hkl"])
            if hkl in seen:
                continue
            seen.add(hkl)
            expanded.append(
                {
                    **reflection,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "star_representative": list(representative),
                    "friedel_mate": bool(mate.get("friedel_mate")),
                }
            )
    expanded.sort(key=lambda item: -float(item.get("relative_structure_factor") or 0.0))
    return expanded, []


def _equivalent_reflections(
    arguments: dict[str, Any],
    structure: dict[str, Any],
    hkl_list: list[list[int]],
    orient1: list[float],
    orient2: list[float],
) -> dict[str, Any]:
    """Equivalent reflections via GSAS-II GenHKLf, flagged for in-plane membership."""
    from tas_copilot_contracts import ToolRuntimeInvocation
    from tas_copilot_core.registry import RuntimeToolRunner
    from tas_copilot_core.structure_factors import _gsasii_runtime_spec

    runtime = _gsasii_runtime_spec(arguments)
    payload = {
        "operation": "equivalent_reflections",
        "structure": {
            "space_group": structure.get("space_group"),
            "space_group_number": structure.get("space_group_number"),
        },
        "hkl_list": hkl_list,
        "gsasii_path": arguments.get("gsasii_path"),
        "gsasii_bin": arguments.get("gsasii_bin"),
    }
    result = RuntimeToolRunner().run_json(
        runtime,
        ToolRuntimeInvocation(runtime_id=runtime.runtime_id, payload=payload),
        tool_name="equivalent_reflections",
        tool_namespace="tas.gsasii",
        tool_version="0.1.0",
    )
    output = result.output or {}
    if result.status.value != "success" or output.get("status") != "success":
        return {"status": "unavailable", "note": "GSAS-II runtime did not return equivalents.", "errors": [error.message for error in result.errors]}
    in_plane = _in_plane_checker(structure.get("lattice") or {}, orient1, orient2)
    rows = []
    for entry in output.get("equivalents") or []:
        full_star: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()
        for mate in entry.get("equivalents") or []:
            for sign in (1, -1):
                signed = tuple(sign * int(value) for value in mate)
                if signed in seen:
                    continue
                seen.add(signed)
                full_star.append({"hkl": list(signed), "in_plane": in_plane(list(signed)) if in_plane else None, "friedel_mate": sign < 0})
        rows.append(
            {
                "hkl": entry.get("hkl"),
                "systematically_absent": entry.get("systematically_absent"),
                "multiplicity": entry.get("multiplicity"),
                "equivalents": full_star,
                "in_plane_equivalents": [item["hkl"] for item in full_star if item["in_plane"]],
            }
        )
    return {"status": "success", "backend": output.get("backend"), "results": rows, "warnings": output.get("warnings") or []}


def _in_plane_checker(lattice: dict[str, float], orient1: list[float], orient2: list[float]):
    try:
        import numpy as np
        from pymatgen.core import Lattice
    except Exception:  # pragma: no cover - optional dependency path.
        return None
    try:
        reciprocal = Lattice.from_parameters(
            lattice["a"], lattice["b"], lattice["c"], lattice["alpha"], lattice["beta"], lattice["gamma"]
        ).reciprocal_lattice
        u1 = np.asarray(reciprocal.get_cartesian_coords(orient1), dtype=float)
        u2 = np.asarray(reciprocal.get_cartesian_coords(orient2), dtype=float)
        normal = np.cross(u1, u2)
        normal /= np.linalg.norm(normal)
    except Exception:
        return None

    def _check(hkl: list[float]) -> bool:
        cart = np.asarray(reciprocal.get_cartesian_coords([float(value) for value in hkl]), dtype=float)
        norm = float(np.linalg.norm(cart))
        if norm <= 0:
            return True
        return abs(float(np.dot(cart, normal))) / norm < 1e-6

    return _check


def _plane_string(orient1: list[float], orient2: list[float]) -> str:
    def _fmt(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:g}" for value in vector) + "]"

    return f"{_fmt(orient1)};{_fmt(orient2)}"


def _bz_section(lattice: dict[str, float], orient1: list[float], orient2: list[float]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Brillouin-zone section through the existing pymatgen-backed helper."""
    try:
        from tas_copilot_core.workflows.check_resolution import _scattering_plane_bz_summary

        summary = _scattering_plane_bz_summary(
            {
                "lattice": {**lattice, "orient1": [orient1], "orient2": [orient2]},
                "q_point": {"h": 0.0, "k": 0.0, "l": 0.0},
                "bz_coordinate_mode": "cartesian_plane",
            }
        )
    except Exception as exc:
        return None, None, [f"Brillouin-zone section unavailable: {type(exc).__name__}: {exc}"]
    return summary.get("bz_section"), summary.get("basis"), []


def _peak_plane_points(suggested: list[dict[str, Any]], lattice: dict[str, float], basis: dict[str, Any]) -> list[dict[str, Any]]:
    """Project suggested peaks (and Friedel mates) into the plane basis."""
    try:
        from pymatgen.core import Lattice
    except Exception:  # pragma: no cover - optional dependency path.
        return []
    reciprocal = Lattice.from_parameters(
        lattice["a"], lattice["b"], lattice["c"], lattice["alpha"], lattice["beta"], lattice["gamma"]
    ).reciprocal_lattice
    e1 = basis["e1_cartesian"]
    e2 = basis["e2_cartesian"]
    points: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for peak in suggested:
        hkl = [float(peak.get("h") or 0), float(peak.get("k") or 0), float(peak.get("l") or 0)]
        for sign in (1.0, -1.0):
            signed = [sign * value for value in hkl]
            key = tuple(signed)
            if key in seen:
                continue
            seen.add(key)
            cart = reciprocal.get_cartesian_coords(signed)
            points.append(
                {
                    "hkl": signed,
                    "label": f"({signed[0]:g} {signed[1]:g} {signed[2]:g})",
                    "x": float(sum(c * e for c, e in zip(cart, e1))),
                    "y": float(sum(c * e for c, e in zip(cart, e2))),
                    "q_inv_angstrom": float(peak.get("q_inv_angstrom") or 0.0),
                    "f2": float(peak.get("relative_structure_factor") or 0.0),
                    "score": peak.get("score"),
                    "plane_status": peak.get("scattering_plane_status"),
                    "peak_id": peak.get("peak_id"),
                    "friedel_mate": sign < 0,
                }
            )
    return points


def _plane_lattice_summary(lattice: dict[str, float], basis: dict[str, Any]) -> dict[str, Any] | None:
    """Reciprocal principal axes and in-plane tiling vectors in plane coordinates.

    - ``reciprocal_axes``: a*, b*, c* projected onto the orthonormal plane
      basis, with the out-of-plane angle so non-orthogonal cells can show
      which principal directions actually lie in (or near) the zone plane.
    - ``tiling``: the two shortest independent reciprocal-lattice vectors that
      lie in the plane, so the Brillouin-zone section can be repeated at every
      in-plane reciprocal-lattice point to cover the displayed area.
    """
    try:
        import numpy as np
        from pymatgen.core import Lattice
    except Exception:  # pragma: no cover - optional dependency path.
        return None
    reciprocal = Lattice.from_parameters(
        lattice["a"], lattice["b"], lattice["c"], lattice["alpha"], lattice["beta"], lattice["gamma"]
    ).reciprocal_lattice
    e1 = np.asarray(basis["e1_cartesian"], dtype=float)
    e2 = np.asarray(basis["e2_cartesian"], dtype=float)
    normal = np.asarray(basis["normal_cartesian"], dtype=float)

    axes = []
    for label, hkl in (("a*", [1, 0, 0]), ("b*", [0, 1, 0]), ("c*", [0, 0, 1])):
        cart = np.asarray(reciprocal.get_cartesian_coords(hkl), dtype=float)
        magnitude = float(np.linalg.norm(cart))
        out_of_plane = math.degrees(math.asin(min(1.0, abs(float(np.dot(cart, normal))) / magnitude))) if magnitude > 0 else 90.0
        axes.append(
            {
                "label": label,
                "hkl": hkl,
                "x": float(np.dot(cart, e1)),
                "y": float(np.dot(cart, e2)),
                "magnitude_inv_angstrom": magnitude,
                "out_of_plane_deg": round(out_of_plane, 3),
                "in_plane": out_of_plane < 1.0,
            }
        )

    candidates: list[dict[str, Any]] = []
    search = 4
    for h in range(-search, search + 1):
        for k in range(-search, search + 1):
            for l in range(-search, search + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                cart = np.asarray(reciprocal.get_cartesian_coords([h, k, l]), dtype=float)
                norm = float(np.linalg.norm(cart))
                if norm < 1e-9 or abs(float(np.dot(cart, normal))) / norm > 1e-6:
                    continue
                candidates.append({"hkl": [h, k, l], "x": float(np.dot(cart, e1)), "y": float(np.dot(cart, e2)), "norm": norm})
    candidates.sort(key=lambda item: item["norm"])
    tiling = None
    warnings: list[str] = []
    if candidates:
        t1 = candidates[0]
        t2 = next(
            (item for item in candidates if abs(t1["x"] * item["y"] - t1["y"] * item["x"]) > 1e-9 * max(t1["norm"] * item["norm"], 1e-12)),
            None,
        )
        if t2 is not None:
            tiling = {
                "t1": {key: t1[key] for key in ("hkl", "x", "y")},
                "t2": {key: t2[key] for key in ("hkl", "x", "y")},
                "note": "Zone sections repeated at in-plane reciprocal-lattice points; sections of zones centered out of this plane are not drawn.",
            }
    if tiling is None:
        warnings.append("No in-plane reciprocal-lattice tiling vectors were found for this plane; only the central Brillouin-zone section is drawn.")
    return {"reciprocal_axes": axes, "tiling": tiling, "warnings": warnings}


def _environment_rings(arguments: dict[str, Any], requested: list[str], *, energy_mev: float, a4_max_deg: float) -> list[dict[str, Any]]:
    """Powder rings for selected environment materials via the same GSAS-II runtime."""
    from tas_copilot_core.structure_factors import _run_gsasii_reflections, _StructureRecord

    ring_sets: list[dict[str, Any]] = []
    for name in requested:
        material = ENVIRONMENT_MATERIALS.get(name)
        if material is None:
            ring_sets.append({"material_id": name, "status": "unknown_material", "rings": [], "warnings": [f"Unknown environment material: {name}"]})
            continue
        lattice = material["lattice"]
        record = _StructureRecord(
            idnum=-1,
            coll_code=None,
            year=None,
            sgr=material["space_group"],
            sgr_num=material["space_group_number"],
            lattice=(lattice["a"], lattice["b"], lattice["c"], lattice["alpha"], lattice["beta"], lattice["gamma"]),
            r_value=None,
            sum_form=material["formula"],
            struct_form=material["formula"],
            chem_name=material["label"],
            atoms=material["atoms"],
            symops=[],
        )
        result = _run_gsasii_reflections(arguments, record, energy_mev=energy_mev, a4_max_deg=a4_max_deg)
        output = result.output or {}
        if result.status.value != "success" or output.get("status") != "success":
            ring_sets.append(
                {
                    "material_id": name,
                    "label": material["label"],
                    "status": "unavailable",
                    "rings": [],
                    "warnings": [f"{material['label']} powder rings unavailable: GSAS-II runtime did not return reflections."],
                }
            )
            continue
        rings = [
            {
                "hkl": [reflection["h"], reflection["k"], reflection["l"]],
                "q_inv_angstrom": reflection["q_inv_angstrom"],
                "d_spacing": reflection["d_spacing"],
                "two_theta_deg": reflection["two_theta_deg"],
                "multiplicity": reflection.get("multiplicity"),
                "relative_structure_factor": reflection.get("relative_structure_factor"),
            }
            for reflection in (output.get("reflections") or [])
        ]
        ring_sets.append(
            {
                "material_id": name,
                "label": material["label"],
                "status": "success",
                "lattice_source": material["source"],
                "rings": rings,
                "warnings": [],
            }
        )
    return ring_sets
