from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import (
    AlignmentPeakCandidate,
    AlignmentPeakSuggestionReport,
    ProvenanceRecord,
    StructureDatabaseCandidate,
    ToolResultStatus,
    ToolRuntimeInvocation,
    ToolRuntimeMode,
    ToolRuntimeSpec,
)

from .registry import RuntimeToolRunner
from .material_identity import parse_formula as parse_material_formula
from .reporting import write_report_bundle


_NEUTRON_B_FM = {
    # Compact common-element table for planning previews. Prefer periodictable
    # when installed; keep this fallback labeled in reports.
    "H": -3.7390,
    "D": 6.671,
    "C": 6.646,
    "O": 5.803,
    "Al": 3.449,
    "Si": 4.1491,
    "P": 5.13,
    "S": 2.847,
    "Ti": -3.438,
    "Mn": -3.73,
    "Fe": 9.45,
    "Ni": 10.3,
    "Cu": 7.718,
    "Sr": 7.02,
    "La": 8.24,
    "Dy": 16.9,
    "Ho": 8.01,
}


@dataclass(frozen=True)
class _StructureRecord:
    idnum: int
    coll_code: int | None
    year: int | None
    sgr: str | None
    sgr_num: int | None
    lattice: tuple[float, float, float, float, float, float]
    r_value: float | None
    sum_form: str
    struct_form: str | None
    chem_name: str | None
    atoms: list[dict[str, Any]]
    symops: list[tuple[list[list[int]], list[float]]]


def default_alignment_peak_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "structure_factors" / "alignment_peak_defaults.yaml"


def default_gsasii_runtime_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "tool_runtimes" / "gsasii.yaml"


def suggest_alignment_peaks(arguments: dict[str, Any]) -> dict[str, Any]:
    formula = str(arguments.get("formula") or "").strip()
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_peaks")).expanduser()
    if not formula:
        return _missing(output_dir, ["formula"], "A chemical formula is required.")

    db_path = arguments.get("icsd_db") or os.environ.get("TAS_ICSD_SQLITE")
    if not db_path:
        return _missing(output_dir, ["icsd_db"], "Provide --icsd-db or TAS_ICSD_SQLITE for the private ICSD SQLite database.")
    db_path = str(Path(str(db_path)).expanduser())
    if not Path(db_path).exists():
        return _missing(output_dir, ["icsd_db"], f"ICSD SQLite database does not exist: {db_path}")

    try:
        target_composition = _parse_formula(formula)
    except Exception as exc:
        report = _base_report(formula, arguments)
        report.update(
            {
                "status": "unavailable",
                "warnings": [f"Formula parsing failed through the optional ASE formula parser: {type(exc).__name__}: {exc}"],
                "missing_inputs": ["parseable_formula"],
            }
        )
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_peak_report.json", markdown_filename="alignment_peak_report.md")
        return report

    config = _load_background_config(arguments.get("background_material_config"))
    background_materials = _background_materials(config, arguments.get("background_material"))

    candidates = _query_icsd_candidates(
        db_path,
        target_composition,
        formula,
        space_group=arguments.get("space_group"),
        space_group_number=_optional_int(arguments.get("space_group_number")),
        limit=max(int(arguments.get("candidate_limit") or 50), int(arguments.get("top_k") or 30)),
    )
    if not candidates:
        report = _base_report(formula, arguments)
        report.update(
            {
                "status": "missing_input",
                "composition": target_composition,
                "warnings": ["No ICSD records matched the normalized composition and optional symmetry filters."],
                "missing_inputs": ["candidate_structure"],
                "background_materials": background_materials,
            }
        )
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_peak_report.json", markdown_filename="alignment_peak_report.md")
        return report

    experiment_lattice = _parse_lattice_argument(arguments.get("lattice"))
    lattice_warnings: list[str] = []
    if experiment_lattice:
        matched = [c for c in candidates if _lattice_matches(experiment_lattice, c.lattice)]
        if matched:
            candidates = matched + [c for c in candidates if c not in matched]
            lattice_warnings.append(
                f"Candidate structures were filtered by the supplied experiment lattice; {len(matched)} of {len(candidates)} match within 3% lengths / 2 deg angles."
            )
        else:
            lattice_warnings.append(
                "No ICSD candidate matches the supplied experiment lattice within 3% lengths / 2 deg angles; "
                "using the formula-ranked best structure. Predicted peak positions may not correspond to the experimenters' cell "
                "(wrong polymorph or setting)."
            )
    selected = candidates[0]
    energy_mev = float(arguments.get("energy_mev") or 14.7)
    top_k = int(arguments.get("top_k") or 30)
    a4_max = float(arguments.get("a4_max_deg") or 120.0)
    warnings = [
        "ICSD content is private/local. Reports contain candidate IDs and derived planning summaries only; do not commit proprietary exports.",
        "Peak suggestions are planning evidence requiring instrument-scientist review, not approved operations.",
    ] + lattice_warnings
    records = _load_structure_records(db_path, [int(selected.source_record_id)])
    selected_record = records.get(int(selected.source_record_id))
    if selected_record is None or not selected_record.atoms:
        report = _alignment_report(
            status="missing_input",
            formula=formula,
            arguments=arguments,
            energy_mev=energy_mev,
            selected=selected,
            candidates=candidates,
            peaks=[],
            background_materials=background_materials,
            warnings=warnings + ["The selected ICSD record did not provide atom positions required for structure-factor calculation."],
            missing_inputs=["atom_positions"],
            db_path=db_path,
            output_dir=output_dir,
        )
        return report

    runtime_result = _run_gsasii_reflections(arguments, selected_record, energy_mev=energy_mev, a4_max_deg=a4_max)
    peaks: list[AlignmentPeakCandidate] = []
    missing_inputs: list[str] = []
    status = "success"
    background_peak_tables: list[dict[str, Any]] = []
    if runtime_result.status != ToolResultStatus.success:
        status = "unavailable" if runtime_result.status == ToolResultStatus.unavailable else "error"
        missing_inputs = ["gsasii_structure_factor_runtime"]
        warnings.extend(error.message for error in runtime_result.errors)
    else:
        adapter_output = runtime_result.output or {}
        adapter_status = adapter_output.get("status")
        warnings.extend(adapter_output.get("warnings") or [])
        if adapter_status != "success":
            status = "missing_input" if adapter_status == "missing_input" else "unavailable"
            missing_inputs = list(adapter_output.get("missing_inputs") or ["gsasii_structure_factor_backend"])
            warnings.extend(error.get("message", str(error)) for error in adapter_output.get("errors", []))
        else:
            background_peak_tables, background_warnings = _background_peak_tables_from_gsasii(
                arguments=arguments,
                db_path=db_path,
                background_materials=background_materials,
                energy_mev=energy_mev,
                a4_max_deg=a4_max,
            )
            warnings.extend(background_warnings)
            peaks, peak_warnings = _build_peak_candidates_from_gsasii_reflections(
                adapter_output.get("reflections") or [],
                selected_record,
                scattering_plane=arguments.get("scattering_plane"),
                top_k=top_k,
                background_materials=background_materials,
                background_peak_tables=background_peak_tables,
                background_overlap_tolerance_deg=float(arguments.get("background_overlap_tolerance_deg") or 0.5),
                low_q_tilt_deg=float(arguments.get("out_of_plane_low_q_deg") or 2.0),
                high_q_tilt_deg=float(arguments.get("out_of_plane_high_q_deg") or 1.0),
                max_out_of_plane_deg=float(arguments.get("out_of_plane_max_deg") or 5.0),
                sample_thickness_mm=_optional_float(arguments.get("sample_thickness_mm")),
            )
            warnings.extend(peak_warnings)
            if not peaks:
                status = "missing_input"
                missing_inputs = ["in_plane_or_allowed_reflections"]
                warnings.append("GSAS-II calculated reflections, but none survived the scattering-plane and A4 filters.")

    return _alignment_report(
        status=status,
        formula=formula,
        arguments=arguments,
        energy_mev=energy_mev,
        selected=selected,
        candidates=candidates,
        peaks=peaks,
        background_materials=background_materials,
        background_peak_tables=background_peak_tables,
        warnings=warnings,
        missing_inputs=missing_inputs,
        db_path=db_path,
        output_dir=output_dir,
    )


def _alignment_report(
    *,
    status: str,
    formula: str,
    arguments: dict[str, Any],
    energy_mev: float,
    selected: StructureDatabaseCandidate,
    candidates: list[StructureDatabaseCandidate],
    peaks: list[AlignmentPeakCandidate],
    background_materials: list[dict[str, Any]],
    background_peak_tables: list[dict[str, Any]] | None = None,
    warnings: list[str],
    missing_inputs: list[str],
    db_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    report_model = AlignmentPeakSuggestionReport(
        status=status,
        formula=formula,
        scattering_plane=arguments.get("scattering_plane"),
        energy_mev=energy_mev,
        selected_structure=selected,
        candidate_structures=candidates,
        alignment_peaks=peaks,
        background_materials=background_materials,
        warnings=warnings,
        missing_inputs=missing_inputs,
        safe_next_actions=_safe_alignment_actions(status, missing_inputs),
        provenance=[
            ProvenanceRecord(
                source_type="private_icsd_sqlite",
                source_id="nist-icsd.sqlite",
                source_uri=db_path,
                method="formula_normalized_icsd_lookup_and_gsasii_peak_generation",
            )
        ],
    )
    report = report_model.model_dump(mode="json")
    report["background_peak_tables"] = background_peak_tables or []
    _write_tables(output_dir, candidates, peaks, background_peak_tables or [])
    report["artifact_paths"] = {
        "candidate_structures_csv": str(output_dir / "candidate_structures.csv"),
        "alignment_peak_table_csv": str(output_dir / "alignment_peak_table.csv"),
        "background_peak_table_csv": str(output_dir / "background_peak_table.csv"),
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_peak_report.json", markdown_filename="alignment_peak_report.md")
    report["artifacts"].update(report["artifact_paths"])
    return report


def _safe_alignment_actions(status: str, missing_inputs: list[str]) -> list[str]:
    actions = [
        "Review the selected ICSD structure before using peak suggestions for an experiment.",
        "Add sample-environment, substrate, mount, cryostat, and analyzer materials with --background-material or a YAML config.",
    ]
    if status != "success" or missing_inputs:
        actions.insert(0, "Configure a GSAS-II runtime or provide the missing structure metadata, then rerun.")
    else:
        actions.append("Use the strongest in-plane peaks as planning candidates, then check instrument limits and backgrounds before measuring.")
    return actions


def _run_gsasii_reflections(arguments: dict[str, Any], record: _StructureRecord, *, energy_mev: float, a4_max_deg: float):
    runtime = _gsasii_runtime_spec(arguments)
    payload = {
        "structure": {
            "source_record_id": record.idnum,
            "space_group": record.sgr,
            "space_group_number": record.sgr_num,
            "space_group_setting": arguments.get("space_group_setting"),
            "lattice": {
                "a": record.lattice[0],
                "b": record.lattice[1],
                "c": record.lattice[2],
                "alpha": record.lattice[3],
                "beta": record.lattice[4],
                "gamma": record.lattice[5],
            },
            "atoms": record.atoms,
        },
        "energy_mev": energy_mev,
        "a4_max_deg": a4_max_deg,
        "max_reflections": int(arguments.get("max_reflections") or 5000),
        "gsasii_path": arguments.get("gsasii_path") or os.environ.get("TAS_GSASII_PATH"),
        "gsasii_bin": arguments.get("gsasii_bin") or os.environ.get("TAS_GSASII_BIN"),
    }
    return RuntimeToolRunner().run_json(
        runtime,
        ToolRuntimeInvocation(runtime_id=runtime.runtime_id, payload=payload),
        tool_name="calculate_structure_factors",
        tool_namespace="tas.gsasii",
        tool_version="0.1.0",
    )


def _gsasii_runtime_spec(arguments: dict[str, Any]) -> ToolRuntimeSpec:
    runtime_path = Path(str(arguments.get("gsasii_runtime") or os.environ.get("TAS_GSASII_RUNTIME") or default_gsasii_runtime_config_path())).expanduser()
    if runtime_path.exists():
        raw = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
        spec = ToolRuntimeSpec.model_validate(raw)
    else:
        spec = ToolRuntimeSpec(
            runtime_id="gsasii-unavailable",
            mode=ToolRuntimeMode.unavailable,
            notes=[f"GSAS-II runtime config not found: {runtime_path}"],
        )
    updates: dict[str, Any] = {}
    if arguments.get("gsasii_python"):
        updates["python_executable"] = str(Path(str(arguments["gsasii_python"])).expanduser())
    env = dict(spec.environment)
    if arguments.get("gsasii_path"):
        env["TAS_GSASII_PATH"] = str(Path(str(arguments["gsasii_path"])).expanduser())
    if arguments.get("gsasii_bin"):
        env["TAS_GSASII_BIN"] = str(Path(str(arguments["gsasii_bin"])).expanduser())
    if updates or env != spec.environment:
        spec = spec.model_copy(update={**updates, "environment": env})
    return spec


def _parse_formula(formula: str) -> dict[str, float]:
    result = parse_material_formula(formula)
    if result.element_amounts:
        return {element: float(amount) for element, amount in result.element_amounts.items()}
    warnings = "; ".join(result.warnings) if result.warnings else "formula parser unavailable or failed"
    raise RuntimeError(warnings)


def _parse_simple_decimal_stoichiometry(formula: str) -> dict[str, float]:
    """Parse simple element-count strings that ASE rejects, such as La0.7Sr0.3MnO3.

    This is intentionally not a general chemistry parser. It only accepts a
    complete sequence of element symbols followed by optional integer or decimal
    amounts, which covers ICSD SUM_FORM strings after whitespace removal and
    common solid-solution formula input.
    """
    if not formula or any(ch in formula for ch in "()[]{}+-"):
        return {}
    token_re = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?|\.\d+)?")
    pos = 0
    amounts: dict[str, float] = {}
    for match in token_re.finditer(formula):
        if match.start() != pos:
            return {}
        element = match.group(1)
        raw_amount = match.group(2)
        amount = float(raw_amount) if raw_amount else 1.0
        amounts[element] = amounts.get(element, 0.0) + amount
        pos = match.end()
    if pos != len(formula):
        return {}
    return amounts


def _formula_matches(candidate_formula: str, target: dict[str, float], *, tolerance: float = 1.0e-6) -> tuple[bool, dict[str, float]]:
    try:
        comp = _parse_formula(candidate_formula)
    except Exception:
        return False, {}
    if set(comp) != set(target):
        return False, comp
    # Allow same reduced formula scale.
    ratios = []
    for el in target:
        if target[el] == 0:
            return False, comp
        ratios.append(comp[el] / target[el])
    if max(ratios) - min(ratios) > tolerance:
        return False, comp
    return True, comp


def _query_icsd_candidates(
    db_path: str,
    target_composition: dict[str, float],
    raw_formula: str,
    *,
    space_group: str | None,
    space_group_number: int | None,
    limit: int,
) -> list[StructureDatabaseCandidate]:
    clauses = ["SUM_FORM is not null"]
    params: list[Any] = []
    for element in sorted(target_composition):
        clauses.append("SUM_FORM like ?")
        params.append(f"%{element}%")
    if space_group_number:
        clauses.append("SGR_NUM = ?")
        params.append(space_group_number)
    elif space_group:
        clauses.append("upper(SGR) = upper(?)")
        params.append(space_group)
    sql = f"""
        select IDNUM, COLL_CODE, YEAR, SGR, SGR_NUM, A_LEN, B_LEN, C_LEN,
               ALPHA, BETA, GAMMA, R_VAL, SUM_FORM, STRUCT_FORM, CHEM_NAME
        from ICSD
        where {' and '.join(clauses)}
        order by YEAR desc, R_VAL asc
        limit ?
    """
    params.append(max(limit * 20, 200))
    rows = sqlite3.connect(db_path).execute(sql, params).fetchall()
    candidates: list[StructureDatabaseCandidate] = []
    for row in rows:
        match, comp = _formula_matches(str(row[12]).replace(" ", ""), target_composition)
        if not match:
            continue
        score_components = _candidate_score_components(row)
        score = round(sum(score_components.values()), 4)
        candidates.append(
            StructureDatabaseCandidate(
                candidate_id=f"icsd:{row[0]}",
                source_database="ICSD",
                source_record_id=str(row[0]),
                formula=raw_formula,
                normalized_formula=str(row[12]),
                composition=comp,
                chemical_name=row[14],
                space_group=row[3],
                space_group_number=row[4],
                lattice={"a": row[5], "b": row[6], "c": row[7], "alpha": row[8], "beta": row[9], "gamma": row[10]},
                year=row[2],
                r_value=row[11],
                score=score,
                score_components=score_components,
                notes=[],
                provenance=[
                    ProvenanceRecord(source_type="private_icsd_sqlite", source_id=str(row[0]), method="normalized_formula_candidate_lookup")
                ],
            )
        )
        if len(candidates) >= limit:
            break
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _parse_lattice_argument(value: Any) -> tuple[float, float, float, float, float, float] | None:
    """Parse an experiment lattice like '5.99 5.99 17.0 90 90 120' (string or list)."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = re.findall(r"-?\d+(?:\.\d+)?", value)
    elif isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        return None
    if len(parts) != 6:
        return None
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _lattice_matches(
    experiment: tuple[float, float, float, float, float, float],
    candidate: dict[str, Any] | None,
    *,
    length_tolerance_fraction: float = 0.03,
    angle_tolerance_deg: float = 2.0,
) -> bool:
    """True when a candidate cell agrees with the experimenters' cell.

    Formula-only ICSD lookup can select the wrong polymorph or setting
    (e.g. beta-RuCl3 P6_3/mcm c~5.7 instead of the honeycomb alpha phase
    c~17); the experiment headers carry the cell the instrument actually
    used, so it is the natural filter.
    """
    if not candidate:
        return False
    try:
        cand = tuple(float(candidate[key]) for key in ("a", "b", "c", "alpha", "beta", "gamma"))
    except (KeyError, TypeError, ValueError):
        return False
    for exp_len, cand_len in zip(experiment[:3], cand[:3]):
        if exp_len <= 0 or abs(exp_len - cand_len) / exp_len > length_tolerance_fraction:
            return False
    for exp_ang, cand_ang in zip(experiment[3:], cand[3:]):
        if abs(exp_ang - cand_ang) > angle_tolerance_deg:
            return False
    return True


def _candidate_score_components(row: tuple[Any, ...]) -> dict[str, float]:
    year = row[2] or 0
    r_value = row[11]
    return {
        "composition_exact": 0.45,
        "has_atom_positions_expected": 0.15,
        "modernity_prior": min(max((float(year) - 1950.0) / 100.0, 0.0), 0.25),
        "r_value_prior": 0.15 if r_value is not None and float(r_value) <= 0.05 else (0.08 if r_value is not None else 0.03),
    }


def _load_structure_records(db_path: str, idnums: list[int]) -> dict[int, _StructureRecord]:
    if not idnums:
        return {}
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in idnums)
    base_rows = conn.execute(
        f"""select IDNUM, COLL_CODE, YEAR, SGR, SGR_NUM, A_LEN, B_LEN, C_LEN,
                  ALPHA, BETA, GAMMA, R_VAL, SUM_FORM, STRUCT_FORM, CHEM_NAME
            from ICSD where IDNUM in ({placeholders})""",
        idnums,
    ).fetchall()
    atoms_by: dict[int, list[dict[str, Any]]] = {i: [] for i in idnums}
    for row in conn.execute(
        f"""select IDNUM, P_SEQ, ATOM_SYMBOL, W_MULT, W_LETT, X_VAL, Y_VAL, Z_VAL, SOF_VAL
            from P_RECORD where IDNUM in ({placeholders}) order by IDNUM, P_SEQ""",
        idnums,
    ):
        atoms_by[int(row[0])].append({"symbol": row[2], "w_mult": row[3], "wyckoff": row[4], "x": row[5], "y": row[6], "z": row[7], "occupancy": row[8] if row[8] is not None else 1.0})
    sgrs = sorted({row[3] for row in base_rows if row[3]})
    sym_by_sgr: dict[str, list[tuple[list[list[int]], list[float]]]] = {}
    for sgr in sgrs:
        sym_by_sgr[sgr] = []
        for row in conn.execute(
            """select T1,R11,R12,R13,T2,R21,R22,R23,T3,R31,R32,R33 from SYMMETRY_RECORDS where SGR=? order by SYM_SEQ""",
            [sgr],
        ):
            sym_by_sgr[sgr].append((
                [[int(row[1]), int(row[2]), int(row[3])], [int(row[5]), int(row[6]), int(row[7])], [int(row[9]), int(row[10]), int(row[11])]],
                [float(row[0]), float(row[4]), float(row[8])],
            ))
    return {
        int(row[0]): _StructureRecord(
            idnum=int(row[0]),
            coll_code=row[1],
            year=row[2],
            sgr=row[3],
            sgr_num=row[4],
            lattice=(float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])),
            r_value=row[11],
            sum_form=str(row[12]),
            struct_form=row[13],
            chem_name=row[14],
            atoms=atoms_by.get(int(row[0]), []),
            symops=sym_by_sgr.get(row[3], []),
        )
        for row in base_rows
    }


def _generate_peak_candidates(
    record: _StructureRecord,
    *,
    energy_mev: float,
    scattering_plane: str | None,
    a4_max_deg: float,
    top_k: int,
    background_materials: list[dict[str, Any]],
    low_q_tilt_deg: float,
    high_q_tilt_deg: float,
    max_out_of_plane_deg: float,
    sample_thickness_mm: float | None,
) -> tuple[list[AlignmentPeakCandidate], list[str]]:
    wavelength = math.sqrt(81.8042096053309 / energy_mev)
    q_max = 4.0 * math.pi * math.sin(math.radians(a4_max_deg / 2.0)) / wavelength
    rec_basis = _reciprocal_basis(record.lattice)
    hmax = max(1, int(math.ceil(q_max / min(_norm(v) for v in rec_basis))) + 1)
    expanded_atoms = _expand_atoms(record)
    b_status = "calculated_common_neutron_b_table"
    missing_b = sorted({atom["symbol"] for atom in expanded_atoms if atom["symbol"] not in _NEUTRON_B_FM})
    warnings = []
    if missing_b:
        b_status = "partial_missing_neutron_b"
        warnings.append(f"Missing neutron scattering lengths for: {', '.join(missing_b)}. Affected F2 values are omitted.")
    rows: list[AlignmentPeakCandidate] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                q_vec = _hkl_vec((h, k, l), rec_basis)
                q = _norm(q_vec)
                if q <= 0 or q > q_max:
                    continue
                angle_arg = q * wavelength / (4.0 * math.pi)
                if angle_arg > 1.0:
                    continue
                two_theta = math.degrees(2.0 * math.asin(angle_arg))
                plane_status, oop = _plane_status((h, k, l), rec_basis, scattering_plane, q, low_q_tilt_deg, high_q_tilt_deg, max_out_of_plane_deg)
                if plane_status == "out_of_plane":
                    continue
                f2 = None if missing_b else _nuclear_f2((h, k, l), expanded_atoms)
                bg = _background_warnings((h, k, l), q, background_materials)
                ms = _multiple_scattering_warning((h, k, l), f2, q, sample_thickness_mm, oop)
                score_components = {
                    "structure_factor": min(math.log10((f2 or 1.0) + 1.0) / 4.0, 0.45) if f2 is not None else 0.0,
                    "in_plane": 0.25 if plane_status == "in_plane" else 0.12,
                    "moderate_angle": 0.15 if 10.0 <= two_theta <= a4_max_deg - 5.0 else 0.05,
                    "background_penalty": -0.08 * len(bg),
                    "multiple_scattering_penalty": -0.08 if ms else 0.0,
                }
                score = max(0.0, round(sum(score_components.values()), 4))
                rows.append(
                    AlignmentPeakCandidate(
                        peak_id=f"hkl:{h},{k},{l}",
                        h=h,
                        k=k,
                        l=l,
                        d_spacing=round(2.0 * math.pi / q, 6),
                        q_inv_angstrom=round(q, 6),
                        two_theta_deg=round(two_theta, 4),
                        scattering_plane_status=plane_status,
                        out_of_plane_angle_deg=round(oop, 4) if oop is not None else None,
                        relative_structure_factor=round(f2, 6) if f2 is not None else None,
                        structure_factor_status=b_status if f2 is not None else "not_calculated",
                        background_warnings=bg,
                        multiple_scattering_warning=ms,
                        score=score,
                        score_components=score_components,
                    )
                )
    rows.sort(key=lambda item: item.score, reverse=True)
    return rows[:top_k], warnings


def _build_peak_candidates_from_gsasii_reflections(
    reflections: list[dict[str, Any]],
    record: _StructureRecord,
    *,
    scattering_plane: str | None,
    top_k: int,
    background_materials: list[dict[str, Any]],
    background_peak_tables: list[dict[str, Any]],
    background_overlap_tolerance_deg: float,
    low_q_tilt_deg: float,
    high_q_tilt_deg: float,
    max_out_of_plane_deg: float,
    sample_thickness_mm: float | None,
) -> tuple[list[AlignmentPeakCandidate], list[str]]:
    rec_basis = _reciprocal_basis(record.lattice)
    rows: list[AlignmentPeakCandidate] = []
    warnings = []
    for reflection in reflections:
        hkl = (int(reflection["h"]), int(reflection["k"]), int(reflection["l"]))
        q = float(reflection["q_inv_angstrom"])
        f2 = float(reflection.get("relative_structure_factor") or 0.0)
        plane_status, oop = _plane_status(hkl, rec_basis, scattering_plane, q, low_q_tilt_deg, high_q_tilt_deg, max_out_of_plane_deg)
        if plane_status == "out_of_plane":
            continue
        bg = _background_warnings(hkl, q, background_materials)
        bg.extend(_background_peak_overlap_warnings(reflection, background_peak_tables, tolerance_deg=background_overlap_tolerance_deg))
        ms = _multiple_scattering_warning(hkl, f2, q, sample_thickness_mm, oop)
        score_components = {
            "structure_factor": min(math.log10(f2 + 1.0) / 4.0, 0.45),
            "in_plane": 0.25 if plane_status == "in_plane" else 0.12,
            "moderate_angle": 0.15 if 10.0 <= float(reflection["two_theta_deg"]) <= 115.0 else 0.05,
            "background_penalty": -0.08 * len(bg),
            "multiple_scattering_penalty": -0.08 if ms else 0.0,
        }
        score = max(0.0, round(sum(score_components.values()), 4))
        rows.append(
            AlignmentPeakCandidate(
                peak_id=f"hkl:{hkl[0]},{hkl[1]},{hkl[2]}",
                h=hkl[0],
                k=hkl[1],
                l=hkl[2],
                d_spacing=round(float(reflection["d_spacing"]), 6),
                q_inv_angstrom=round(q, 6),
                two_theta_deg=round(float(reflection["two_theta_deg"]), 4),
                scattering_plane_status=plane_status,
                out_of_plane_angle_deg=round(oop, 4) if oop is not None else None,
                relative_structure_factor=round(f2, 6),
                structure_factor_status=str(reflection.get("structure_factor_status") or "calculated_gsasii"),
                background_warnings=bg,
                multiple_scattering_warning=ms,
                score=score,
                score_components=score_components,
                provenance=[
                    ProvenanceRecord(
                        source_type="external_runtime",
                        source_id="GSAS-II",
                        method="GSASIIstrMath.StructureFactor2",
                        metadata={"source_record_id": record.idnum},
                    )
                ],
            )
        )
    rows.sort(key=lambda item: item.score, reverse=True)
    if reflections and not rows:
        warnings.append("All GSAS-II reflections were filtered out by scattering-plane constraints.")
    return rows[:top_k], warnings


def _expand_atoms(record: _StructureRecord) -> list[dict[str, Any]]:
    if not record.symops:
        return record.atoms
    expanded = []
    seen: set[tuple[str, int, int, int]] = set()
    tol = 1.0e-5
    for atom in record.atoms:
        xyz = [float(atom["x"] or 0.0), float(atom["y"] or 0.0), float(atom["z"] or 0.0)]
        for rot, trans in record.symops:
            pos = [sum(rot[i][j] * xyz[j] for j in range(3)) + trans[i] for i in range(3)]
            pos = [p % 1.0 for p in pos]
            key = (str(atom["symbol"]), *(int(round(p / tol)) for p in pos))
            if key in seen:
                continue
            seen.add(key)
            new_atom = dict(atom)
            new_atom.update({"x": pos[0], "y": pos[1], "z": pos[2]})
            expanded.append(new_atom)
    return expanded


def _nuclear_f2(hkl: tuple[int, int, int], atoms: list[dict[str, Any]]) -> float:
    real = 0.0
    imag = 0.0
    h, k, l = hkl
    for atom in atoms:
        b = _NEUTRON_B_FM.get(str(atom["symbol"]))
        if b is None:
            continue
        occ = float(atom.get("occupancy") if atom.get("occupancy") is not None else 1.0)
        phase = 2.0 * math.pi * (h * float(atom["x"]) + k * float(atom["y"]) + l * float(atom["z"]))
        real += occ * b * math.cos(phase)
        imag += occ * b * math.sin(phase)
    return real * real + imag * imag


def _reciprocal_basis(cell: tuple[float, float, float, float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    a, b, c, alpha, beta, gamma = cell
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    av = (a, 0.0, 0.0)
    bv = (b * math.cos(gr), b * math.sin(gr), 0.0)
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    cv = (cx, cy, cz)
    vol = _dot(av, _cross(bv, cv))
    return (
        _scale(_cross(bv, cv), 2.0 * math.pi / vol),
        _scale(_cross(cv, av), 2.0 * math.pi / vol),
        _scale(_cross(av, bv), 2.0 * math.pi / vol),
    )


def _hkl_vec(hkl: tuple[float, float, float] | tuple[int, int, int], basis: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    return tuple(sum(hkl[j] * basis[j][i] for j in range(3)) for i in range(3))


def _plane_status(hkl: tuple[int, int, int], basis: tuple[tuple[float, float, float], ...], plane: str | None, q: float, low: float, high: float, max_angle: float = 5.0) -> tuple[str, float | None]:
    if not plane:
        return "not_filtered", None
    basis_hkls = _plane_basis_hkls(plane)
    if not basis_hkls:
        return "unknown_plane", None
    # Use the reciprocal-space metric tensor, matching the scalar/modvec
    # convention in rescalculator/lattice_calculator.py for latticestar.
    # HKL index space is not Euclidean for general cells.
    gstar = _reciprocal_metric_from_basis(basis)
    q_metric = _metric_norm(hkl, gstar)
    residual = _metric_distance_to_span(hkl, basis_hkls, gstar)
    angle = math.degrees(math.asin(min(1.0, residual / q_metric))) if q_metric else 90.0
    threshold = min(max_angle, max(high, min(low, low - (low - high) * min(q_metric / 6.0, 1.0))))
    if angle < 0.05:
        return "in_plane", angle
    if angle <= threshold:
        return "near_plane_vertical_acceptance", angle
    if angle <= max_angle:
        return "near_plane_projection_candidate", angle
    return "out_of_plane", angle


def _reciprocal_metric_from_basis(basis: tuple[tuple[float, float, float], ...]) -> list[list[float]]:
    return [[_dot(basis[i], basis[j]) for j in range(3)] for i in range(3)]


def _metric_dot_hkl(a: tuple[float, float, float] | tuple[int, int, int], b: tuple[float, float, float] | tuple[int, int, int], metric: list[list[float]]) -> float:
    return sum(float(a[i]) * float(b[j]) * metric[i][j] for i in range(3) for j in range(3))


def _metric_norm(vec: tuple[float, float, float] | tuple[int, int, int], metric: list[list[float]]) -> float:
    return math.sqrt(max(_metric_dot_hkl(vec, vec, metric), 0.0))


def _metric_distance_to_span(
    vec: tuple[float, float, float] | tuple[int, int, int],
    span: list[tuple[float, float, float]],
    metric: list[list[float]],
) -> float:
    if not span:
        return _metric_norm(vec, metric)
    if len(span) == 1:
        u = span[0]
        uu = _metric_dot_hkl(u, u, metric)
        if uu < 1.0e-12:
            return _metric_norm(vec, metric)
        uv = _metric_dot_hkl(vec, u, metric)
        dist2 = _metric_dot_hkl(vec, vec, metric) - uv * uv / uu
        return math.sqrt(max(dist2, 0.0))
    u, v = span[0], span[1]
    uu = _metric_dot_hkl(u, u, metric)
    uv = _metric_dot_hkl(u, v, metric)
    vv = _metric_dot_hkl(v, v, metric)
    bu = _metric_dot_hkl(vec, u, metric)
    bv = _metric_dot_hkl(vec, v, metric)
    det = uu * vv - uv * uv
    if abs(det) < 1.0e-12:
        return _metric_distance_to_span(vec, [u], metric)
    coeff_u = (bu * vv - bv * uv) / det
    coeff_v = (uu * bv - uv * bu) / det
    proj_dot = coeff_u * bu + coeff_v * bv
    dist2 = _metric_dot_hkl(vec, vec, metric) - proj_dot
    return math.sqrt(max(dist2, 0.0))


def _plane_basis_hkls(plane: str) -> list[tuple[float, float, float]]:
    explicit = _explicit_plane_basis_hkls(plane)
    if explicit:
        return explicit
    chars = [c.upper() for c in re.sub(r"[^A-Za-z0]", "", plane)]
    if len(chars) != 3:
        return []
    variables = sorted({c for c in chars if c in {"H", "K", "L"}})
    basis = []
    for var in variables:
        basis.append(tuple(float(1 if c == var else 0) for c in chars))  # type: ignore[arg-type]
    return basis


def _explicit_plane_basis_hkls(plane: str) -> list[tuple[float, float, float]]:
    groups: list[tuple[float, float, float]] = []
    bracketed = re.findall(r"[\[\(]\s*([^\]\)]+?)\s*[\]\)]", plane)
    candidates = bracketed if bracketed else re.split(r";", plane)
    for candidate in candidates:
        parts = re.findall(r"-?\d+(?:\.\d+)?", candidate)
        if len(parts) >= 3:
            groups.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return groups[:2] if len(groups) >= 2 else []


def _distance_to_span(vec: tuple[float, float, float], span: list[tuple[float, float, float]]) -> float:
    if len(span) == 1:
        u = span[0]
        proj = _scale(u, _dot(vec, u) / max(_dot(u, u), 1.0e-12))
    else:
        n = _cross(span[0], span[1])
        nn = _dot(n, n)
        if nn < 1.0e-12:
            return _distance_to_span(vec, span[:1])
        return abs(_dot(vec, n)) / math.sqrt(nn)
    return _norm(_sub(vec, proj))


def _background_warnings(hkl: tuple[int, int, int], q: float, materials: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for material in materials:
        if material.get("enabled_by_default") or material.get("enabled"):
            label = material.get("label") or material.get("material_id")
            if label == "Aluminum" and 2.5 <= q <= 2.9:
                warnings.append("Aluminum background peak family may be nearby; review mount/cryostat geometry.")
    return warnings


def _multiple_scattering_warning(hkl: tuple[int, int, int], f2: float | None, q: float, thickness: float | None, oop: float | None) -> str | None:
    if f2 is None or f2 < 100.0:
        return None
    if thickness is None:
        return "multiple-scattering heuristic: strong Bragg candidate; sample thickness/mosaic missing, so risk is unknown."
    if thickness >= 1.0 and (oop or 0.0) <= 2.0:
        return "multiple-scattering heuristic: strong Bragg candidate with finite thickness and near-plane acceptance."
    return None


def _background_peak_tables_from_gsasii(
    *,
    arguments: dict[str, Any],
    db_path: str,
    background_materials: list[dict[str, Any]],
    energy_mev: float,
    a4_max_deg: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    tables: list[dict[str, Any]] = []
    warnings: list[str] = []
    max_background_reflections = int(arguments.get("max_background_reflections") or 300)
    for material in background_materials:
        if not (material.get("enabled_by_default") or material.get("enabled") or material.get("source") == "user_input"):
            continue
        formula = str(material.get("formula") or material.get("label") or "").strip()
        if not formula:
            warnings.append(f"Background material {material.get('label') or material.get('material_id')} has no formula; peak table not generated.")
            continue
        try:
            composition = _parse_formula(formula)
        except Exception as exc:
            warnings.append(f"Background material {material.get('label') or formula} formula was not parseable: {type(exc).__name__}: {exc}")
            continue
        candidates = _query_icsd_candidates(
            db_path,
            composition,
            formula,
            space_group=material.get("space_group"),
            space_group_number=_optional_int(material.get("space_group_number")),
            limit=3,
        )
        if not candidates:
            warnings.append(f"No ICSD background structure matched {material.get('label') or formula}; no background peak table generated.")
            continue
        records = _load_structure_records(db_path, [int(candidates[0].source_record_id)])
        record = records.get(int(candidates[0].source_record_id))
        if record is None or not record.atoms:
            warnings.append(f"Background structure {candidates[0].source_record_id} for {material.get('label') or formula} lacks atom positions.")
            continue
        runtime_result = _run_gsasii_reflections({**arguments, "max_reflections": max_background_reflections}, record, energy_mev=energy_mev, a4_max_deg=a4_max_deg)
        if runtime_result.status != ToolResultStatus.success:
            warnings.extend(f"Background {material.get('label') or formula}: {error.message}" for error in runtime_result.errors)
            continue
        adapter_output = runtime_result.output or {}
        if adapter_output.get("status") != "success":
            warnings.append(f"Background {material.get('label') or formula}: GSAS-II adapter status {adapter_output.get('status')}.")
            warnings.extend(error.get("message", str(error)) for error in adapter_output.get("errors", []))
            continue
        reflections = []
        for row in adapter_output.get("reflections") or []:
            reflections.append(
                {
                    "material_id": material.get("material_id"),
                    "material_label": material.get("label") or formula,
                    "role": material.get("role"),
                    "h": int(row["h"]),
                    "k": int(row["k"]),
                    "l": int(row["l"]),
                    "hkl": [int(row["h"]), int(row["k"]), int(row["l"])],
                    "d_spacing": row.get("d_spacing"),
                    "q_inv_angstrom": row.get("q_inv_angstrom"),
                    "two_theta_deg": row.get("two_theta_deg"),
                    "relative_structure_factor": row.get("relative_structure_factor"),
                    "structure_factor_status": row.get("structure_factor_status") or "calculated_gsasii",
                }
            )
        tables.append(
            {
                "material_id": material.get("material_id"),
                "label": material.get("label") or formula,
                "formula": formula,
                "role": material.get("role"),
                "status": "success",
                "selected_structure": candidates[0].model_dump(mode="json"),
                "reflection_count": len(reflections),
                "reflections": reflections,
                "provenance": [
                    ProvenanceRecord(
                        source_type="private_icsd_sqlite",
                        source_id=str(candidates[0].source_record_id),
                        method="background_material_gsasii_structure_factor_table",
                    ).model_dump(mode="json")
                ],
            }
        )
    return tables, warnings


def _background_peak_overlap_warnings(reflection: dict[str, Any], tables: list[dict[str, Any]], *, tolerance_deg: float) -> list[str]:
    two_theta = reflection.get("two_theta_deg")
    if two_theta is None or not tables:
        return []
    observed = float(two_theta)
    warnings = []
    for table in tables:
        for row in table.get("reflections") or []:
            other = row.get("two_theta_deg")
            if other is None:
                continue
            delta = abs(observed - float(other))
            if delta <= tolerance_deg:
                label = row.get("material_label") or table.get("label") or row.get("material_id")
                hkl = row.get("hkl") or [row.get("h"), row.get("k"), row.get("l")]
                warnings.append(
                    f"GSAS-II background overlap candidate: {label} ({hkl[0]} {hkl[1]} {hkl[2]}) at two-theta {float(other):.3f} deg, delta {delta:.3f} deg."
                )
    return warnings


def _load_background_config(path: str | None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else default_alignment_peak_config_path()
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _background_materials(config: dict[str, Any], extra: str | list[str] | None) -> list[dict[str, Any]]:
    materials = [dict(item) for item in config.get("background_materials", [])]
    extras = extra if isinstance(extra, list) else ([extra] if extra else [])
    for item in extras:
        materials.append({"material_id": str(item).lower().replace(" ", "_"), "label": str(item), "formula": str(item), "enabled": True, "source": "user_input"})
    return materials


def _write_tables(output_dir: Path, candidates: list[StructureDatabaseCandidate], peaks: list[AlignmentPeakCandidate], background_peak_tables: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate_structures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "source_record_id", "year", "space_group", "space_group_number", "a", "b", "c", "r_value", "score", "formula"])
        for item in candidates:
            lat = item.lattice
            writer.writerow([item.candidate_id, item.source_record_id, item.year, item.space_group, item.space_group_number, lat.get("a"), lat.get("b"), lat.get("c"), item.r_value, item.score, item.normalized_formula])
    with (output_dir / "alignment_peak_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["h", "k", "l", "d_spacing", "q_inv_angstrom", "two_theta_deg", "plane_status", "out_of_plane_angle_deg", "relative_f2", "score", "warnings"])
        for item in peaks:
            writer.writerow([item.h, item.k, item.l, item.d_spacing, item.q_inv_angstrom, item.two_theta_deg, item.scattering_plane_status, item.out_of_plane_angle_deg, item.relative_structure_factor, item.score, "; ".join(item.background_warnings + ([item.multiple_scattering_warning] if item.multiple_scattering_warning else []))])
    with (output_dir / "background_peak_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id", "label", "role", "h", "k", "l", "d_spacing", "q_inv_angstrom", "two_theta_deg", "relative_f2"])
        for table in background_peak_tables:
            for item in table.get("reflections") or []:
                writer.writerow([table.get("material_id"), table.get("label"), table.get("role"), item.get("h"), item.get("k"), item.get("l"), item.get("d_spacing"), item.get("q_inv_angstrom"), item.get("two_theta_deg"), item.get("relative_structure_factor")])


def _base_report(formula: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"workflow_name": "suggest-alignment-peaks", "formula": formula, "scattering_plane": arguments.get("scattering_plane"), "energy_mev": arguments.get("energy_mev")}


def _missing(output_dir: Path, fields: list[str], message: str) -> dict[str, Any]:
    report = {"workflow_name": "suggest-alignment-peaks", "status": "missing_input", "missing_inputs": fields, "warnings": [message], "safe_next_actions": ["Provide the missing input and rerun the workflow."]}
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_peak_report.json", markdown_filename="alignment_peak_report.md")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Alignment Peak Suggestions", "", f"Status: `{report.get('status')}`", ""]
    if report.get("formula"):
        lines.append(f"- Formula: `{report.get('formula')}`")
    if report.get("scattering_plane"):
        lines.append(f"- Scattering plane: `{report.get('scattering_plane')}`")
    if report.get("energy_mev"):
        lines.append(f"- Energy: `{report.get('energy_mev')}` meV")
    selected = report.get("selected_structure") or {}
    if selected:
        lines.extend(["", "## Selected Candidate Structure", "", f"- ICSD record: `{selected.get('source_record_id')}`", f"- Formula: `{selected.get('normalized_formula')}`", f"- Space group: `{selected.get('space_group')}` ({selected.get('space_group_number')})", f"- Score: `{selected.get('score')}`"])
    lines.extend(["", "## Warnings"])
    for warning in report.get("warnings", []):
        lines.append(f"- {warning}")
    bg_tables = report.get("background_peak_tables") or []
    if bg_tables:
        lines.extend(["", "## Background Peak Tables", ""])
        for table in bg_tables:
            lines.append(f"- `{table.get('label')}` ({table.get('formula')}): `{table.get('reflection_count')}` GSAS-II reflections")
    lines.extend(["", "## Top Peaks", "", "| HKL | Q (1/A) | two-theta (deg) | plane status | rel. F2 | score | warning |", "|---|---:|---:|---|---:|---:|---|"])
    for peak in report.get("alignment_peaks", [])[:20]:
        warn = "; ".join((peak.get("background_warnings") or []) + ([peak.get("multiple_scattering_warning")] if peak.get("multiple_scattering_warning") else []))
        lines.append(f"| ({peak.get('h')} {peak.get('k')} {peak.get('l')}) | {peak.get('q_inv_angstrom')} | {peak.get('two_theta_deg')} | {peak.get('scattering_plane_status')} | {peak.get('relative_structure_factor')} | {peak.get('score')} | {warn} |")
    lines.extend(["", "## Safe Conclusion", "", "These are candidate alignment peaks from a selected candidate structure and configured background materials. They are planning suggestions requiring experimenter review, not approved operations."])
    return "\n".join(lines) + "\n"


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))
