from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any


def main() -> int:
    import contextlib

    try:
        payload = json.load(sys.stdin)
        # GSAS-II prints first-run config banners to stdout, which would
        # corrupt the JSON-on-stdout protocol; route library prints to stderr
        # and emit only the JSON result on stdout.
        with contextlib.redirect_stdout(sys.stderr):
            if payload.get("operation") == "equivalent_reflections":
                result = equivalent_reflections(payload)
            else:
                result = calculate_reflections(payload)
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        result = {
            "status": "error",
            "errors": [{"code": "gsasii_adapter_error", "message": f"{type(exc).__name__}: {exc}"}],
            "reflections": [],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


def _configure_import_paths(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    path = payload.get("gsasii_path") or os.environ.get("TAS_GSASII_PATH")
    binary = payload.get("gsasii_bin") or os.environ.get("TAS_GSASII_BIN")
    if path:
        path = os.path.expanduser(str(path))
        binary = binary or os.path.join(os.path.dirname(path), "bin")
        # Support both legacy modules and package-style GSAS-II installs.
        for candidate in (binary, path, os.path.dirname(path.rstrip("/"))):
            if candidate and candidate not in sys.path:
                sys.path.insert(0, str(candidate))
    elif binary:
        sys.path.insert(0, os.path.expanduser(str(binary)))
    return path, binary


def equivalent_reflections(payload: dict[str, Any]) -> dict[str, Any]:
    """Symmetry-equivalent reflections for given HKLs through GSAS-II GenHKLf."""
    gsasii_path, gsasii_bin = _configure_import_paths(payload)
    try:
        try:
            import GSASIIspc as G2spc
        except ImportError:
            from GSASII import GSASIIspc as G2spc
    except Exception as exc:
        return {
            "status": "unavailable",
            "errors": [{"code": "gsasii_import_failed", "message": f"{type(exc).__name__}: {exc}"}],
            "equivalents": [],
            "backend": {"name": "GSAS-II", "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }
    structure = payload.get("structure") or {}
    space_group = _space_group_candidates(structure.get("space_group"), structure.get("space_group_number"), structure.get("space_group_setting"))
    try:
        sg_number = int(structure.get("space_group_number") or 0)
        by_num_table = getattr(G2spc, "spgbyNum", None)
        if by_num_table and 0 < sg_number < len(by_num_table) and by_num_table[sg_number]:
            space_group.append(str(by_num_table[sg_number]))
    except Exception:
        pass
    sg_data = None
    sg_used = None
    sg_error = None
    for sg in space_group:
        try:
            err, data = G2spc.SpcGroup(sg)
        except Exception as exc:
            sg_error = f"{type(exc).__name__}: {exc}"
            continue
        if not err:
            sg_data = data
            sg_used = data.get("SpGrp", sg)
            break
        sg_error = str(err)
    if sg_data is None:
        return {
            "status": "missing_input",
            "errors": [{"code": "space_group_unusable", "message": f"GSAS-II could not parse space group {structure.get('space_group')!r}: {sg_error}"}],
            "equivalents": [],
        }
    results = []
    for hkl in payload.get("hkl_list") or []:
        h, k, l = (int(value) for value in hkl)
        absent, multiplicity, unique, _phi = G2spc.GenHKLf([h, k, l], sg_data)
        results.append(
            {
                "hkl": [h, k, l],
                "systematically_absent": bool(absent),
                "multiplicity": int(multiplicity),
                "equivalents": [[int(u[0]), int(u[1]), int(u[2])] for u in unique],
            }
        )
    return {
        "status": "success",
        "backend": {"name": "GSAS-II", "space_group": sg_used, "method": "GSASIIspc.GenHKLf"},
        "equivalents": results,
        "warnings": ["Equivalents are point-symmetry mates from GSAS-II; Friedel pairs are included via the -hkl set where applicable."],
    }


def calculate_reflections(payload: dict[str, Any]) -> dict[str, Any]:
    gsasii_path, gsasii_bin = _configure_import_paths(payload)

    try:
        import numpy as np
        try:
            import GSASIIElem as G2el
            import GSASIIlattice as G2lat
            import GSASIIspc as G2spc
            import GSASIIstrMath as G2str
        except ImportError:
            from GSASII import GSASIIElem as G2el
            from GSASII import GSASIIlattice as G2lat
            from GSASII import GSASIIspc as G2spc
            from GSASII import GSASIIstrMath as G2str
    except Exception as exc:
        return {
            "status": "unavailable",
            "errors": [{"code": "gsasii_import_failed", "message": f"{type(exc).__name__}: {exc}"}],
            "reflections": [],
            "backend": {"name": "GSAS-II", "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }

    structure = payload.get("structure") or {}
    lattice = structure.get("lattice") or {}
    atoms = structure.get("atoms") or []
    space_group = _space_group_candidates(structure.get("space_group"), structure.get("space_group_number"), structure.get("space_group_setting"))
    # GSAS-II ships its own IT-number -> canonical-symbol table; use it as the
    # generic fallback so pymatgen/CIF symbol spellings (e.g. P6_3/mmc) do not
    # have to match GSAS-II's spacing conventions symbol-by-symbol.
    try:
        sg_number = int(structure.get("space_group_number") or 0)
        by_num_table = getattr(G2spc, "spgbyNum", None)
        if by_num_table and 0 < sg_number < len(by_num_table) and by_num_table[sg_number]:
            space_group.append(str(by_num_table[sg_number]))
    except Exception:
        pass
    if not atoms:
        return _missing("atoms", "No atom records were supplied to the GSAS-II adapter.", gsasii_path, gsasii_bin)
    try:
        cell = [
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            float(lattice.get("alpha", 90.0)),
            float(lattice.get("beta", 90.0)),
            float(lattice.get("gamma", 90.0)),
        ]
    except Exception:
        return _missing("lattice", "Complete lattice parameters are required.", gsasii_path, gsasii_bin)

    sg_error = None
    sg_data = None
    sg_used = None
    for sg in space_group:
        try:
            err, data = G2spc.SpcGroup(sg)
        except Exception as exc:
            sg_error = f"{type(exc).__name__}: {exc}"
            continue
        if not err:
            sg_data = data
            sg_used = data.get("SpGrp", sg)
            break
        sg_error = str(err)
    if sg_data is None:
        return {
            "status": "missing_input",
            "errors": [{"code": "space_group_unusable", "message": f"GSAS-II could not parse space group {structure.get('space_group')!r}: {sg_error}"}],
            "reflections": [],
            "backend": {"name": "GSAS-II", "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }

    energy_mev = float(payload.get("energy_mev") or 14.7)
    wavelength = math.sqrt(81.8042096053309 / energy_mev)
    a4_max = float(payload.get("a4_max_deg") or 120.0)
    d_min = wavelength / (2.0 * math.sin(math.radians(a4_max / 2.0)))
    max_reflections = int(payload.get("max_reflections") or 5000)

    atom_types = sorted({_element_symbol(atom.get("symbol")) for atom in atoms if _element_symbol(atom.get("symbol"))})
    if not atom_types:
        return _missing("atom_types", "No valid atom element symbols were supplied.", gsasii_path, gsasii_bin)
    try:
        bl_table = G2el.GetBLtable({"AtomTypes": atom_types, "Isotope": {el: "Nat. Abund." for el in atom_types}})
    except Exception as exc:
        return {
            "status": "unavailable",
            "errors": [{"code": "neutron_scattering_lengths_unavailable", "message": f"{type(exc).__name__}: {exc}"}],
            "reflections": [],
            "backend": {"name": "GSAS-II", "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }

    a_tensor = G2lat.cell2A(cell)
    g_tensor, _direct_g = G2lat.A2Gmat(a_tensor)
    hkl_d = G2lat.GenHLaue(d_min, sg_data, a_tensor)
    if not hkl_d:
        return {
            "status": "missing_input",
            "errors": [{"code": "no_reflections", "message": "GSAS-II generated no reflections in the requested A4 range."}],
            "reflections": [],
            "backend": {"name": "GSAS-II", "space_group": sg_used, "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }

    ref_rows = []
    for h, k, l, d_spacing in hkl_d[:max_reflections]:
        ext, mul, _uniq, _phi = G2spc.GenHKLf([int(h), int(k), int(l)], sg_data)
        if ext:
            continue
        ref_rows.append([h, k, l, mul * 2, d_spacing, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    if not ref_rows:
        return {
            "status": "missing_input",
            "errors": [{"code": "no_non_extinct_reflections", "message": "All generated reflections were extinct for the selected space group."}],
            "reflections": [],
            "backend": {"name": "GSAS-II", "space_group": sg_used, "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }

    pfx = "0::"
    hfx = ":0:"
    phfx = "0:0:"
    calc_controls = {
        "FFtables": {},
        "EFtables": {},
        "BLtables": bl_table,
        "ORBtables": {},
        hfx + "histType": "PNC",
        "Natoms": {pfx: len(atoms)},
    }
    parm_dict = {phfx + "BabA": 0.0, phfx + "BabU": 0.0, hfx + "Lam": wavelength}
    for idx, atom in enumerate(atoms):
        symbol = _element_symbol(atom.get("symbol"))
        parm_dict[pfx + f"Atype:{idx}"] = symbol
        parm_dict[pfx + f"Amul:{idx}"] = float(atom.get("w_mult") or atom.get("multiplicity") or 1.0)
        parm_dict[pfx + f"Afrac:{idx}"] = float(atom.get("occupancy") if atom.get("occupancy") is not None else 1.0)
        parm_dict[pfx + f"Ax:{idx}"] = float(atom.get("x") or 0.0)
        parm_dict[pfx + f"Ay:{idx}"] = float(atom.get("y") or 0.0)
        parm_dict[pfx + f"Az:{idx}"] = float(atom.get("z") or 0.0)
        parm_dict[pfx + f"AUiso:{idx}"] = float(atom.get("uiso") if atom.get("uiso") is not None else 0.01)

    ref_dict = {"RefList": np.array(ref_rows, dtype=float), "FF": {}}
    try:
        G2str.StructureFactor2(ref_dict, g_tensor, hfx, pfx, sg_data, calc_controls, parm_dict)
    except Exception as exc:
        return {
            "status": "error",
            "errors": [{"code": "structure_factor_failed", "message": f"{type(exc).__name__}: {exc}"}],
            "reflections": [],
            "backend": {"name": "GSAS-II", "space_group": sg_used, "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
        }

    reflections = []
    for row in ref_dict["RefList"]:
        h, k, l = int(row[0]), int(row[1]), int(row[2])
        d_spacing = float(row[4])
        q = 2.0 * math.pi / d_spacing
        angle_arg = min(1.0, max(0.0, wavelength / (2.0 * d_spacing)))
        two_theta = math.degrees(2.0 * math.asin(angle_arg))
        if two_theta > a4_max + 1.0e-9:
            continue
        reflections.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "d_spacing": d_spacing,
                "q_inv_angstrom": q,
                "two_theta_deg": two_theta,
                "multiplicity": float(row[3]),
                "relative_structure_factor": float(row[9]),
                "structure_factor_status": "calculated_gsasii_neutron_nuclear",
            }
        )

    reflections.sort(key=lambda item: item["relative_structure_factor"], reverse=True)
    return {
        "status": "success",
        "backend": {
            "name": "GSAS-II",
            "space_group": sg_used,
            "gsasii_path": gsasii_path,
            "gsasii_bin": gsasii_bin,
            "histogram_type": "PNC",
            "wavelength_angstrom": wavelength,
        },
        "reflections": reflections,
        "warnings": ["Calculated neutron nuclear F^2 through GSAS-II StructureFactor2 using natural-abundance scattering lengths."],
    }


def _missing(field: str, message: str, gsasii_path: str, gsasii_bin: str) -> dict[str, Any]:
    return {
        "status": "missing_input",
        "missing_inputs": [field],
        "errors": [{"code": f"missing_{field}", "message": message}],
        "reflections": [],
        "backend": {"name": "GSAS-II", "gsasii_path": gsasii_path, "gsasii_bin": gsasii_bin},
    }


def _element_symbol(value: Any) -> str:
    match = re.match(r"([A-Z][a-z]?)", str(value or "").strip())
    return match.group(1) if match else ""


def _space_group_candidates(value: Any, number: Any = None, setting: Any = None) -> list[str]:
    raw = str(value or "P 1").strip()
    candidates = [str(setting).strip()] if setting else []
    candidates.append(raw)
    if "_" in raw:
        candidates.append(raw.replace("_", ""))
    compact = raw.replace(" ", "").upper()
    by_number = {
        1: "P 1",
        140: "I 4/m c m",
        167: "R -3 c H",
        221: "P m -3 m",
        225: "F m -3 m",
        227: "F d -3 m",
    }
    try:
        if int(number) in by_number:
            candidates.append(by_number[int(number)])
    except Exception:
        pass
    if compact == "P1":
        candidates.append("P 1")
    if compact == "R-3C":
        candidates.extend(["R -3 c H", "R -3 c"])
    if compact == "PM-3M":
        candidates.append("P m -3 m")
    if compact == "FM-3M":
        candidates.append("F m -3 m")
    if compact == "FD-3M":
        candidates.append("F d -3 m")
    if compact == "I4/MCM":
        candidates.append("I 4/m c m")
    return list(dict.fromkeys(candidates))


if __name__ == "__main__":
    raise SystemExit(main())
