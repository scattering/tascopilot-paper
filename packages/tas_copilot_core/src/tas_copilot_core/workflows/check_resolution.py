from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ToolCall, ToolResult
from tas_copilot_contracts.tools import ToolResultStatus

from .common import default_runner


def check_resolution(arguments: dict) -> ToolResult:
    viz_keys = {"output_dir", "write_plotly", "scan_points", "bz_coordinate_mode"}
    call_arguments = {key: value for key, value in arguments.items() if key not in viz_keys}
    viz_arguments = dict(call_arguments)
    if "bz_coordinate_mode" in arguments:
        viz_arguments["bz_coordinate_mode"] = arguments["bz_coordinate_mode"]
    scan_points = arguments.get("scan_points") or []
    runner = default_runner()
    call = ToolCall(tool_namespace="tas.resolution", tool_name="calculate_resolution", arguments=call_arguments, caller_workflow="check-resolution")
    result = runner.run(call)
    if scan_points and result.status == ToolResultStatus.success:
        scan_records = [_scan_record_from_result(call_arguments, result, point_index=0)]
        for index, point in enumerate(scan_points, start=1):
            point_arguments = _merge_scan_point_arguments(call_arguments, point)
            point_call = ToolCall(tool_namespace="tas.resolution", tool_name="calculate_resolution", arguments=point_arguments, caller_workflow="check-resolution")
            point_result = runner.run(point_call)
            if point_result.status != ToolResultStatus.success:
                result.warnings.append(f"scan point {index} resolution calculation skipped: {point_result.status}")
                continue
            scan_records.append(_scan_record_from_result(point_arguments, point_result, point_index=index))
            result.provenance.extend(point_result.provenance)
        if result.output is not None:
            result.output["resolution_scan"] = {
                "point_count": len(scan_records),
                "points": scan_records,
                "summary": "Resolution scan calculated point-by-point through the registered resolution tool.",
            }
    output_dir = arguments.get("output_dir")
    if output_dir and result.status == ToolResultStatus.success:
        artifacts = write_resolution_plotly_artifacts(result, Path(output_dir), viz_arguments)
        result.derived_artifacts.extend(artifacts)
    return result


def write_resolution_plotly_artifacts(result: ToolResult, output_dir: Path, arguments: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Write browser-native resolution ellipse artifacts from a resolution matrix."""

    scan_points = (((result.output or {}).get("resolution_scan") or {}).get("points") or [])
    if scan_points:
        records = scan_points
    else:
        matrix = ((result.output or {}).get("resolution") or {}).get("resolution_matrix")
        if not matrix:
            result.warnings.append("resolution Plotly visualization skipped: no resolution_matrix in tool output")
            return []
        records = [{"point_index": 0, "q_point": {}, "energy_transfer_meV": 0.0, "center": {"qx_inv_angstrom": 0.0, "qy_inv_angstrom": 0.0, "energy_meV": 0.0}, "resolution_matrix": matrix}]
    if not records:
        result.warnings.append("resolution Plotly visualization skipped: no resolution_matrix in tool output")
        return []
    try:
        panels = _resolution_ellipse_sections_for_records(records, _resolution_axis_conversions(arguments or {}))
    except Exception as exc:
        result.warnings.append(f"resolution Plotly visualization skipped: {type(exc).__name__}: {exc}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "success",
        "visualization_type": "plotly_resolution_projection_and_cross_section_ellipses",
        "coordinate_system": "rescalculator Qx/Qy/E convention",
        "contour": "half_maximum_gaussian_contour",
        "projection_semantics": {
            "projection": "Integrated/projected 2D ellipse using the Schur complement, matching rescalculator ResPlot.project(...)[0].",
            "cross_section": "Direct 2D slice using the selected submatrix, matching rescalculator ResPlot.project(...)[1].",
        },
        "point_count": len(records),
        "source_tool": f"{result.tool_namespace}.{result.tool_name}",
        "audit_id": result.audit_id,
        "input_hash": result.input_hash,
        "output_hash": result.output_hash,
        "panels": panels,
    }
    json_path = output_dir / "resolution_ellipse_projections.json"
    html_path = output_dir / "resolution_ellipse_projections.html"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_resolution_plotly_html(summary), encoding="utf-8")
    artifacts = [
        {
            "artifact_type": "resolution_plotly_json",
            "path": str(json_path),
            "description": "Resolution projection and cross-section ellipse coordinates for browser rendering.",
            "lineage": {"input_hash": result.input_hash, "output_hash": result.output_hash, "audit_id": result.audit_id},
        },
        {
            "artifact_type": "resolution_plotly_html",
            "path": str(html_path),
            "description": "Interactive Plotly resolution ellipse projection/cross-section report.",
            "lineage": {"input_hash": result.input_hash, "output_hash": result.output_hash, "audit_id": result.audit_id},
        },
    ]
    if arguments:
        artifacts.extend(_write_scattering_plane_bz_artifacts(arguments, output_dir, result))
    return artifacts


def _merge_scan_point_arguments(base: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in (point or {}).items():
        if key == "q_point":
            merged["q_point"] = value
        elif key == "energy_transfer":
            merged["energy_transfer"] = value
        elif key == "energy_transfer_meV":
            merged["energy_transfer"] = {"value": {"value": float(value), "unit": "meV"}}
        else:
            merged[key] = value
    return merged


def _scan_record_from_result(arguments: dict[str, Any], result: ToolResult, *, point_index: int) -> dict[str, Any]:
    matrix = ((result.output or {}).get("resolution") or {}).get("resolution_matrix")
    q_point = arguments.get("q_point") or {}
    hkl = [float(q_point.get("h", 0.0)), float(q_point.get("k", 0.0)), float(q_point.get("l", 0.0))]
    energy = _energy_mev(arguments.get("energy_transfer"))
    center = _scattering_plane_center(arguments, hkl, energy)
    return {
        "point_index": point_index,
        "q_point": {"h": hkl[0], "k": hkl[1], "l": hkl[2]},
        "energy_transfer_meV": energy,
        "center": center,
        "resolution_matrix": matrix,
        "audit_id": result.audit_id,
        "input_hash": result.input_hash,
        "output_hash": result.output_hash,
    }


def _energy_mev(value: Any) -> float:
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, dict):
            return float(nested.get("value", 0.0))
        if nested is not None:
            return float(nested)
    if value is not None:
        return float(value)
    return 0.0


def _scattering_plane_center(arguments: dict[str, Any], hkl: list[float], energy: float) -> dict[str, float]:
    try:
        import numpy as np
        from pymatgen.core import Lattice

        lattice_args = arguments.get("lattice") or {}
        lattice = Lattice.from_parameters(
            float(lattice_args["a"]),
            float(lattice_args["b"]),
            float(lattice_args["c"]),
            float(lattice_args["alpha"]),
            float(lattice_args["beta"]),
            float(lattice_args["gamma"]),
        )
        reciprocal = lattice.reciprocal_lattice
        orient1 = _first_vector(lattice_args.get("orient1") or [1, 0, 0])
        orient2 = _first_vector(lattice_args.get("orient2") or [0, 1, 0])
        u1 = np.asarray(reciprocal.get_cartesian_coords(orient1), dtype=float)
        u2 = np.asarray(reciprocal.get_cartesian_coords(orient2), dtype=float)
        e1 = u1 / np.linalg.norm(u1)
        u2_perp = u2 - np.dot(u2, e1) * e1
        e2 = u2_perp / np.linalg.norm(u2_perp)
        q_cart = np.asarray(reciprocal.get_cartesian_coords(hkl), dtype=float)
        return {"qx_inv_angstrom": float(np.dot(q_cart, e1)), "qy_inv_angstrom": float(np.dot(q_cart, e2)), "energy_meV": float(energy)}
    except Exception:
        return {"qx_inv_angstrom": 0.0, "qy_inv_angstrom": 0.0, "energy_meV": float(energy)}


def _resolution_axis_conversions(arguments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        import numpy as np
        from pymatgen.core import Lattice

        lattice_args = arguments.get("lattice") or {}
        lattice = Lattice.from_parameters(
            float(lattice_args["a"]),
            float(lattice_args["b"]),
            float(lattice_args["c"]),
            float(lattice_args["alpha"]),
            float(lattice_args["beta"]),
            float(lattice_args["gamma"]),
        )
        reciprocal = lattice.reciprocal_lattice
        orient1 = _first_vector(lattice_args.get("orient1") or [1, 0, 0])
        orient2 = _first_vector(lattice_args.get("orient2") or [0, 1, 0])
        u1 = np.asarray(reciprocal.get_cartesian_coords(orient1), dtype=float)
        u2 = np.asarray(reciprocal.get_cartesian_coords(orient2), dtype=float)
        qx_scale = float(np.linalg.norm(u1))
        if qx_scale < 1e-12:
            raise ValueError("orient1 maps to a near-zero reciprocal vector")
        e1 = u1 / qx_scale
        u2_perp = u2 - np.dot(u2, e1) * e1
        qy_scale = float(np.linalg.norm(u2_perp))
        if qy_scale < 1e-12:
            raise ValueError("orient1 and orient2 are collinear in reciprocal space")
        return {
            "qx": {
                "scale_inv_angstrom_per_basis_unit": qx_scale,
                "axis_title": f"Qx units of [{orient1[0]:g} {orient1[1]:g} {orient1[2]:g}]",
                "basis_hkl": orient1,
            },
            "qy": {
                "scale_inv_angstrom_per_basis_unit": qy_scale,
                "axis_title": f"Qy units of projected [{orient2[0]:g} {orient2[1]:g} {orient2[2]:g}]",
                "basis_hkl": orient2,
            },
        }
    except Exception:
        return {
            "qx": {"scale_inv_angstrom_per_basis_unit": 1.0, "axis_title": "Qx basis units", "basis_hkl": [1.0, 0.0, 0.0]},
            "qy": {"scale_inv_angstrom_per_basis_unit": 1.0, "axis_title": "Qy basis units", "basis_hkl": [0.0, 1.0, 0.0]},
        }


def _resolution_ellipse_sections_for_records(records: list[dict[str, Any]], axis_conversions: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    axis_conversions = axis_conversions or _resolution_axis_conversions({})
    panels = [
        ("Qx_Qy", 0, 1, 2, "Qx (1/A)", "Qy (1/A)", "qx_inv_angstrom", "qy_inv_angstrom", "qx", "qy"),
        ("Qx_E", 0, 2, 1, "Qx (1/A)", "E (meV)", "qx_inv_angstrom", "energy_meV", "qx", None),
        ("Qy_E", 1, 2, 0, "Qy (1/A)", "E (meV)", "qy_inv_angstrom", "energy_meV", "qy", None),
    ]
    output: list[dict[str, Any]] = []
    for panel_id, i, j, eliminated, x_axis, y_axis, x_center_key, y_center_key, x_secondary_key, y_secondary_key in panels:
        ellipses: list[dict[str, Any]] = []
        centers_x: list[float] = []
        centers_y: list[float] = []
        for record in records:
            matrix = record.get("resolution_matrix")
            if not matrix:
                continue
            center = record.get("center") or {}
            x0 = float(center.get(x_center_key, 0.0))
            y0 = float(center.get(y_center_key, 0.0))
            centered = _resolution_ellipse_sections_for_matrix(matrix, i, j, eliminated, x0=x0, y0=y0)
            for ellipse in centered:
                ellipse["point_index"] = record.get("point_index")
                ellipse["q_point"] = record.get("q_point")
            ellipses.extend(centered)
            centers_x.append(x0)
            centers_y.append(y0)
        output.append(
            {
                "panel_id": panel_id,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "x_secondary_axis": _secondary_axis(axis_conversions, x_secondary_key),
                "y_secondary_axis": _secondary_axis(axis_conversions, y_secondary_key),
                "eliminated_axis": ["Qx", "Qy", "E", "Qz"][eliminated],
                "scan_path": {"x": centers_x, "y": centers_y},
                "ellipses": ellipses,
            }
        )
    return output


def _secondary_axis(axis_conversions: dict[str, dict[str, Any]], key: str | None) -> dict[str, Any] | None:
    if key is None:
        return None
    entry = dict(axis_conversions.get(key) or {})
    scale = float(entry.get("scale_inv_angstrom_per_basis_unit") or 1.0)
    if scale <= 0:
        scale = 1.0
    return {
        "axis_title": entry.get("axis_title") or f"{key} basis units",
        "scale_inv_angstrom_per_basis_unit": scale,
        "basis_hkl": entry.get("basis_hkl") or [],
        "conversion": "basis_units = inverse_angstrom / scale_inv_angstrom_per_basis_unit",
    }


def _resolution_ellipse_sections_for_matrix(matrix: list[list[float]], i: int, j: int, eliminated: int, *, x0: float, y0: float) -> list[dict[str, Any]]:
    import numpy as np

    precision = np.asarray(matrix, dtype=float)
    if precision.shape != (4, 4):
        raise ValueError(f"expected a 4x4 resolution matrix, got {precision.shape}")
    projected = _schur_projection_matrix(precision, i, j, eliminated)
    section = precision[np.ix_([i, j], [i, j])]
    return [
        {
            "ellipse_type": "projection",
            "label": "projection / integrated",
            "line_dash": "solid",
            "color": "#b91c1c",
            **_ellipse_trace_from_precision(projected, x0=x0, y0=y0),
        },
        {
            "ellipse_type": "cross_section",
            "label": "cross-section / slice",
            "line_dash": "dash",
            "color": "#1d4ed8",
            **_ellipse_trace_from_precision(section, x0=x0, y0=y0),
        },
    ]


def _schur_projection_matrix(matrix, i: int, j: int, eliminated: int):
    import numpy as np

    keep = [i, j]
    base = matrix[np.ix_(keep, keep)]
    coupling = matrix[np.ix_(keep, [eliminated])]
    denom = matrix[eliminated, eliminated]
    if abs(float(denom)) < 1e-15:
        raise ValueError(f"cannot project through near-zero diagonal for axis {eliminated}")
    return base - coupling @ coupling.T / denom


def _ellipse_trace_from_precision(matrix, *, x0: float = 0.0, y0: float = 0.0) -> dict[str, Any]:
    import numpy as np

    precision = np.asarray(matrix, dtype=float)
    if precision.shape != (2, 2):
        raise ValueError(f"expected a 2x2 ellipse matrix, got {precision.shape}")
    phi = np.linspace(0.0, 2.0 * math.pi, 361)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    denominator = precision[0, 0] * cos_phi * cos_phi + precision[1, 1] * sin_phi * sin_phi + 2.0 * precision[0, 1] * cos_phi * sin_phi
    denominator = np.asarray(denominator, dtype=float)
    if np.any(denominator <= 0):
        raise ValueError("ellipse matrix is not positive definite in all sampled directions")
    radius = np.sqrt(2.0 * math.log(2.0) / denominator)
    values, vectors = np.linalg.eigh(precision)
    values = np.maximum(values, 1e-300)
    hwhm = np.sqrt(2.0 * math.log(2.0) / values)
    return {
        "matrix": [[float(value) for value in row] for row in precision.tolist()],
        "hwhm_axes": [float(value) for value in hwhm],
        "center": [float(x0), float(y0)],
        "x": [float(value) for value in (radius * cos_phi + x0)],
        "y": [float(value) for value in (radius * sin_phi + y0)],
    }


def _write_scattering_plane_bz_artifacts(arguments: dict[str, Any], output_dir: Path, result: ToolResult) -> list[dict[str, Any]]:
    try:
        summary = _scattering_plane_bz_summary(arguments)
    except ImportError as exc:
        result.warnings.append(f"scattering-plane BZ Plotly visualization skipped: optional dependency missing: {exc}")
        return []
    except Exception as exc:
        result.warnings.append(f"scattering-plane BZ Plotly visualization skipped: {type(exc).__name__}: {exc}")
        return []
    summary.update(
        {
            "source_tool": f"{result.tool_namespace}.{result.tool_name}",
            "audit_id": result.audit_id,
            "input_hash": result.input_hash,
            "output_hash": result.output_hash,
        }
    )
    json_path = output_dir / "scattering_plane_bz_pymatgen.json"
    html_path = output_dir / "scattering_plane_bz_pymatgen.html"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_scattering_plane_bz_html(summary), encoding="utf-8")
    return [
        {
            "artifact_type": "scattering_plane_bz_plotly_json",
            "path": str(json_path),
            "description": "pymatgen-derived Brillouin-zone section in the TAS scattering plane.",
            "lineage": {"input_hash": result.input_hash, "output_hash": result.output_hash, "audit_id": result.audit_id},
        },
        {
            "artifact_type": "scattering_plane_bz_plotly_html",
            "path": str(html_path),
            "description": "Interactive Plotly scattering-plane Brillouin-zone prototype.",
            "lineage": {"input_hash": result.input_hash, "output_hash": result.output_hash, "audit_id": result.audit_id},
        },
    ]


def _scattering_plane_bz_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from pymatgen.core import Lattice

    lattice_args = arguments.get("lattice") or {}
    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    missing = [key for key in required if key not in lattice_args]
    if missing:
        raise ValueError(f"missing lattice fields: {', '.join(missing)}")
    orient1 = _first_vector(lattice_args.get("orient1") or [1, 0, 0])
    orient2 = _first_vector(lattice_args.get("orient2") or [0, 1, 0])
    lattice = Lattice.from_parameters(
        float(lattice_args["a"]),
        float(lattice_args["b"]),
        float(lattice_args["c"]),
        float(lattice_args["alpha"]),
        float(lattice_args["beta"]),
        float(lattice_args["gamma"]),
    )
    reciprocal = lattice.reciprocal_lattice
    u1 = np.asarray(reciprocal.get_cartesian_coords(orient1), dtype=float)
    u2 = np.asarray(reciprocal.get_cartesian_coords(orient2), dtype=float)
    norm1 = float(np.linalg.norm(u1))
    if norm1 < 1e-12:
        raise ValueError("orient1 maps to a near-zero reciprocal vector")
    e1 = u1 / norm1
    u2_perp = u2 - np.dot(u2, e1) * e1
    norm2 = float(np.linalg.norm(u2_perp))
    if norm2 < 1e-12:
        raise ValueError("orient1 and orient2 are collinear in reciprocal space")
    e2 = u2_perp / norm2
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal)
    faces = reciprocal.get_wigner_seitz_cell()
    points = _plane_polyhedron_intersections(faces, normal)
    if len(points) < 3:
        raise ValueError("scattering plane did not produce a polygonal BZ section")
    q_point = arguments.get("q_point") or {}
    q_hkl = [float(q_point.get("h", 0.0)), float(q_point.get("k", 0.0)), float(q_point.get("l", 0.0))]
    q_cart = np.asarray(reciprocal.get_cartesian_coords(q_hkl), dtype=float)
    cartesian_polygon, cartesian_q = _ordered_plane_coordinates(points, q_cart, e1, e2)
    hkl_polygon, hkl_q = _ordered_hkl_basis_coordinates(points, q_cart, u1, u2)
    coordinate_modes = {
        "cartesian_plane": {
            "label": "orthonormal scattering-plane basis",
            "x_axis": "orient1 direction (1/A)",
            "y_axis": "orthogonalized orient2 direction (1/A)",
            "units": "1/Angstrom with pymatgen reciprocal-lattice 2pi convention",
            "bz_section": {
                "x": [float(value) for value in cartesian_polygon[:, 0]],
                "y": [float(value) for value in cartesian_polygon[:, 1]],
            },
            "q_point": {"hkl": q_hkl, "plane_coordinates": [float(value) for value in cartesian_q]},
        },
        "hkl_basis": {
            "label": "non-orthogonal HKL scattering-plane basis",
            "x_axis": f"coefficient along [{orient1[0]:g} {orient1[1]:g} {orient1[2]:g}]",
            "y_axis": f"coefficient along [{orient2[0]:g} {orient2[1]:g} {orient2[2]:g}]",
            "units": "basis coefficients in the TAS scattering-plane HKL vectors",
            "warning": "Axes are generally non-orthogonal; distances and angles should be interpreted through the reciprocal metric, not Euclidean screen geometry.",
            "bz_section": {
                "x": [float(value) for value in hkl_polygon[:, 0]],
                "y": [float(value) for value in hkl_polygon[:, 1]],
            },
            "q_point": {"hkl": q_hkl, "plane_coordinates": [float(value) for value in hkl_q]},
        },
    }
    selected_mode = str(arguments.get("bz_coordinate_mode") or "cartesian_plane")
    if selected_mode not in coordinate_modes:
        selected_mode = "cartesian_plane"
    selected = coordinate_modes[selected_mode]
    return {
        "status": "success",
        "visualization_type": "pymatgen_scattering_plane_bz_section",
        "coordinate_system": selected["label"],
        "selected_coordinate_mode": selected_mode,
        "available_coordinate_modes": list(coordinate_modes),
        "units": selected["units"],
        "lattice": {key: float(lattice_args[key]) for key in required},
        "orient1_hkl": [float(value) for value in orient1],
        "orient2_hkl": [float(value) for value in orient2],
        "basis": {
            "e1_cartesian": [float(value) for value in e1],
            "e2_cartesian": [float(value) for value in e2],
            "normal_cartesian": [float(value) for value in normal],
        },
        "coordinate_modes": coordinate_modes,
        "bz_section": selected["bz_section"],
        "q_point": selected["q_point"],
    }


def _ordered_plane_coordinates(points: list[Any], q_cart: Any, e1: Any, e2: Any):
    import numpy as np

    projected = np.asarray([[float(np.dot(point, e1)), float(np.dot(point, e2))] for point in points], dtype=float)
    polygon = _close_ordered_polygon(projected)
    q_plot = np.asarray([float(np.dot(q_cart, e1)), float(np.dot(q_cart, e2))], dtype=float)
    return polygon, q_plot


def _ordered_hkl_basis_coordinates(points: list[Any], q_cart: Any, u1: Any, u2: Any):
    import numpy as np

    basis = np.column_stack([u1, u2])
    projected = np.asarray([np.linalg.lstsq(basis, np.asarray(point, dtype=float), rcond=None)[0] for point in points], dtype=float)
    polygon = _close_ordered_polygon(projected)
    q_plot = np.linalg.lstsq(basis, np.asarray(q_cart, dtype=float), rcond=None)[0]
    return polygon, q_plot


def _close_ordered_polygon(projected):
    import numpy as np

    center = projected.mean(axis=0)
    order = np.argsort(np.arctan2(projected[:, 1] - center[1], projected[:, 0] - center[0]))
    polygon = projected[order]
    return np.vstack([polygon, polygon[0]])


def _first_vector(value: Any) -> list[float]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("orientation vectors must have three components")
    return [float(item) for item in value]


def _plane_polyhedron_intersections(faces: list[list[Any]], normal) -> list[Any]:
    import numpy as np

    candidates: list[Any] = []
    seen_edges: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    tol = 1e-9
    for face in faces:
        vertices = [np.asarray(vertex, dtype=float) for vertex in face]
        for index, p0 in enumerate(vertices):
            p1 = vertices[(index + 1) % len(vertices)]
            key0 = tuple(round(float(value), 10) for value in p0)
            key1 = tuple(round(float(value), 10) for value in p1)
            edge_key = tuple(sorted([key0, key1]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            d0 = float(np.dot(normal, p0))
            d1 = float(np.dot(normal, p1))
            if abs(d0) <= tol:
                candidates.append(p0)
            if abs(d1) <= tol:
                candidates.append(p1)
            if d0 * d1 < -tol * tol:
                t = d0 / (d0 - d1)
                candidates.append(p0 + t * (p1 - p0))
    unique: list[Any] = []
    seen_points: set[tuple[float, float, float]] = set()
    for point in candidates:
        key = tuple(round(float(value), 9) for value in point)
        if key in seen_points:
            continue
        seen_points.add(key)
        unique.append(point)
    return unique


def _resolution_plotly_html(summary: dict[str, Any]) -> str:
    payload = json.dumps(summary)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resolution Ellipse Projections</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    .note {{ color: #52616f; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .panel {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 10px; }}
    code {{ background: #eef2f6; padding: 0.1rem 0.25rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Resolution Ellipse Projections</h1>
  <p class="note">Read-only visualization of half-maximum Gaussian contour projections from the registered resolution tool output. Red solid curves are projected/integrated ellipses; blue dashed curves are direct cross-section/slice ellipses. This follows the rescalculator <code>project()</code> convention.</p>
  <p class="note">Bottom/left axes are inverse angstrom or meV. Top axes convert reciprocal-space coordinates into units of the plotted scattering-plane basis vector; the right axis is added for the <code>Qx-Qy</code> panel.</p>
  <p class="note">Audit: <code id="audit"></code> Input hash: <code id="input-hash"></code></p>
  <div id="plots" class="grid"></div>
  <script>
    const report = {payload};
    document.getElementById('audit').textContent = report.audit_id || '';
    document.getElementById('input-hash').textContent = report.input_hash || '';
    const container = document.getElementById('plots');
    for (const panelDef of report.panels || []) {{
      const panel = document.createElement('div');
      panel.className = 'panel';
      const plot = document.createElement('div');
      plot.style.height = '360px';
      panel.appendChild(plot);
      container.appendChild(panel);
      const traces = (panelDef.ellipses || []).map(ellipse => ({{
        x: ellipse.x,
        y: ellipse.y,
        mode: 'lines',
        type: 'scatter',
        name: `${{ellipse.label}} ${{ellipse.point_index ?? ''}}`,
        line: {{color: ellipse.color || '#205493', width: 2, dash: ellipse.line_dash || 'solid'}}
      }}));
      traces.unshift({{
        x: panelDef.scan_path?.x || [],
        y: panelDef.scan_path?.y || [],
        mode: 'lines+markers',
        type: 'scatter',
        name: 'scan path',
        line: {{color: '#111827', width: 1}},
        marker: {{color: '#111827', size: 5}}
      }});
      const xs = traces.flatMap(trace => trace.x || []).filter(Number.isFinite);
      const ys = traces.flatMap(trace => trace.y || []).filter(Number.isFinite);
      const rangeWithPad = values => {{
        if (!values.length) return undefined;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const pad = Math.max((max - min) * 0.08, 1e-6);
        return [min - pad, max + pad];
      }};
      const xRange = rangeWithPad(xs);
      const yRange = rangeWithPad(ys);
      const layout = {{
        title: panelDef.panel_id,
        xaxis: {{title: panelDef.x_axis, zeroline: true, range: xRange}},
        yaxis: {{title: panelDef.y_axis, zeroline: true, range: yRange}},
        margin: {{l: 70, r: panelDef.y_secondary_axis ? 80 : 25, t: panelDef.x_secondary_axis ? 85 : 55, b: 60}}
      }};
      if (panelDef.x_secondary_axis && xRange) {{
        const scale = panelDef.x_secondary_axis.scale_inv_angstrom_per_basis_unit || 1;
        layout.xaxis2 = {{
          title: panelDef.x_secondary_axis.axis_title,
          overlaying: 'x',
          side: 'top',
          range: [xRange[0] / scale, xRange[1] / scale],
          showgrid: false,
          zeroline: false
        }};
      }}
      if (panelDef.y_secondary_axis && yRange) {{
        const scale = panelDef.y_secondary_axis.scale_inv_angstrom_per_basis_unit || 1;
        layout.yaxis2 = {{
          title: panelDef.y_secondary_axis.axis_title,
          overlaying: 'y',
          side: 'right',
          range: [yRange[0] / scale, yRange[1] / scale],
          showgrid: false,
          zeroline: false
        }};
      }}
      Plotly.newPlot(plot, traces, {{...layout}}, {{responsive: true}}).then(() => {{
        plot.on('plotly_relayout', event => {{
          const update = {{}};
          if (panelDef.x_secondary_axis && Number.isFinite(event['xaxis.range[0]']) && Number.isFinite(event['xaxis.range[1]'])) {{
            const scale = panelDef.x_secondary_axis.scale_inv_angstrom_per_basis_unit || 1;
            update['xaxis2.range'] = [event['xaxis.range[0]'] / scale, event['xaxis.range[1]'] / scale];
          }}
          if (panelDef.y_secondary_axis && Number.isFinite(event['yaxis.range[0]']) && Number.isFinite(event['yaxis.range[1]'])) {{
            const scale = panelDef.y_secondary_axis.scale_inv_angstrom_per_basis_unit || 1;
            update['yaxis2.range'] = [event['yaxis.range[0]'] / scale, event['yaxis.range[1]'] / scale];
          }}
          if (Object.keys(update).length) Plotly.relayout(plot, update);
        }});
      }});
    }}
  </script>
</body>
</html>
"""


def _scattering_plane_bz_html(summary: dict[str, Any]) -> str:
    payload = json.dumps(summary)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scattering Plane Brillouin-Zone Section</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    .note {{ color: #52616f; }}
    code {{ background: #eef2f6; padding: 0.1rem 0.25rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Scattering Plane Brillouin-Zone Section</h1>
  <p class="note">Prototype browser view. Pymatgen builds the reciprocal lattice and Wigner-Seitz Brillouin zone; tascopilot intersects that polyhedron with the TAS scattering plane defined by <code>orient1</code> and <code>orient2</code>. This is a planning visualization, not an approved instrument operation.</p>
  <label>Coordinate mode <select id="mode"></select></label>
  <p id="mode-note" class="note"></p>
  <p class="note">Audit: <code id="audit"></code> Input hash: <code id="input-hash"></code></p>
  <div id="plot" style="height:560px;"></div>
  <script>
    const report = {payload};
    document.getElementById('audit').textContent = report.audit_id || '';
    document.getElementById('input-hash').textContent = report.input_hash || '';
    const select = document.getElementById('mode');
    const modes = report.coordinate_modes || {{}};
    for (const [key, mode] of Object.entries(modes)) {{
      const option = document.createElement('option');
      option.value = key;
      option.textContent = `${{key}} - ${{mode.label || ''}}`;
      select.appendChild(option);
    }}
    select.value = report.selected_coordinate_mode || 'cartesian_plane';
    function drawMode(modeKey) {{
      const mode = modes[modeKey] || modes.cartesian_plane || {{}};
      const bz = mode.bz_section || {{}};
      const q = mode.q_point || {{}};
      document.getElementById('mode-note').textContent = [mode.units || '', mode.warning || ''].filter(Boolean).join(' ');
      const lockAspect = modeKey !== 'hkl_basis';
      Plotly.newPlot('plot', [
        {{
          x: bz.x || [],
          y: bz.y || [],
          mode: 'lines',
          fill: 'toself',
          type: 'scatter',
          name: 'BZ section',
          line: {{color: '#205493', width: 2}},
          fillcolor: 'rgba(32,84,147,0.08)'
        }},
        {{
          x: [0],
          y: [0],
          mode: 'markers',
          type: 'scatter',
          name: 'Gamma',
          marker: {{color: '#111827', size: 8}}
        }},
        {{
          x: [q.plane_coordinates?.[0]],
          y: [q.plane_coordinates?.[1]],
          mode: 'markers',
          type: 'scatter',
          name: 'query Q',
          marker: {{color: '#b91c1c', size: 10}}
        }}
      ], {{
        title: `Scattering plane: orient1=${{JSON.stringify(report.orient1_hkl)}} orient2=${{JSON.stringify(report.orient2_hkl)}}`,
        xaxis: {{title: mode.x_axis || 'x', zeroline: true}},
        yaxis: {{title: mode.y_axis || 'y', zeroline: true, ...(lockAspect ? {{scaleanchor: 'x', scaleratio: 1}} : {{}})}},
        margin: {{l: 70, r: 20, t: 70, b: 70}}
      }}, {{responsive: true}});
    }}
    select.addEventListener('change', () => drawMode(select.value));
    drawMode(select.value);
  </script>
</body>
</html>
"""
