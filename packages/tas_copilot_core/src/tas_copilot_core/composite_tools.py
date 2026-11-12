from __future__ import annotations

from tas_copilot_contracts import Availability, ImplementationType, PermissionLevel, SideEffectLevel, ToolResult, ToolResultStatus, ToolSpec

from tas_copilot_core.material_identity import compare_material_identity, material_identity_record, parse_formula, parse_material_text
from tas_copilot_core.workflows.composite import apply_corrections, combine_files, read_bt7_data


def get_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="read_bt7_data",
            namespace="tas.data_reader",
            version="0.1.0",
            description="Compatibility workflow that classifies a TAS file, reads TAS metadata, and attempts BT-7-specific parsing through registry tools.",
            input_schema={"type": "object", "required": ["path"]},
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.none,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["tas-data-readers and/or bt7grouping plugins"],
            preconditions=["A local file path is supplied."],
            postconditions=["No source files are modified."],
            provenance_policy="Record subtool calls and audit ids.",
            permissions_required=[PermissionLevel.read_only],
            enabled_by_default=True,
            owner_repo="https://github.com/scattering/tascopilot-paper",
        ),
        ToolSpec(
            name="combine_files",
            namespace="tas.file_arithmetic",
            version="0.1.0",
            description="Compatibility workflow that dispatches to explicit file arithmetic tools such as add_scans or subtract_background.",
            input_schema={"type": "object", "required": ["operation"]},
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.local_files,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["tasreduction plugin"],
            preconditions=["operation is add_scans or subtract_background."],
            postconditions=["Derived artifacts, if any, are produced by external plugins with lineage."],
            provenance_policy="Record selected operation, subtool call, and audit id.",
            permissions_required=[PermissionLevel.writes_local_files],
            enabled_by_default=True,
            owner_repo="https://github.com/scattering/tascopilot-paper",
        ),
        ToolSpec(
            name="apply_corrections",
            namespace="tas.corrections",
            version="0.1.0",
            description="Correction-pipeline workflow over explicit correction tools such as normalize_to_monitor and apply_detailed_balance.",
            input_schema={"type": "object", "required": ["steps"]},
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.local_files,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["tasreduction plugin"],
            preconditions=["steps contains supported correction tool names."],
            postconditions=["No correction internals are implemented in core."],
            provenance_policy="Record correction steps, subtool calls, and audit ids.",
            permissions_required=[PermissionLevel.writes_local_files],
            enabled_by_default=True,
            owner_repo="https://github.com/scattering/tascopilot-paper",
        ),
        ToolSpec(
            name="parse_formula",
            namespace="tas.material_identity",
            version="0.1.0",
            description="Parse standard chemical formulas through optional chemistry backends, preferring pymatgen and falling back to ASE where available.",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}, "preferred_backend": {"type": "string"}},
            },
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.none,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["optional pymatgen and/or ASE"],
            preconditions=["Formula text is supplied."],
            postconditions=["No source files are modified. Missing parser support is reported as unavailable output, not guessed chemistry."],
            provenance_policy="Record source text, parser backend, parser version, and interpretation method.",
            permissions_required=[PermissionLevel.read_only],
            enabled_by_default=True,
            owner_repo="https://github.com/scattering/tascopilot-paper",
        ),
        ToolSpec(
            name="parse_filename_composition",
            namespace="tas.material_identity",
            version="0.1.0",
            description="Interpret TAS filename/sample-log composition shorthands using curated family grammars, preserving ambiguity.",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}, "family": {"type": "string"}},
            },
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.none,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["core curated family grammar"],
            preconditions=["A curated material family grammar exists."],
            postconditions=["Filename-derived material identity evidence is returned with ambiguity warnings where needed."],
            provenance_policy="Record source text, grammar match method, parser backend, and ambiguity reason.",
            permissions_required=[PermissionLevel.read_only],
            enabled_by_default=True,
            owner_repo="https://github.com/scattering/tascopilot-paper",
        ),
        ToolSpec(
            name="compare_material_identity",
            namespace="tas.material_identity",
            version="0.1.0",
            description="Compare two material identity strings using resolved composition or curated variable-composition grammar evidence.",
            input_schema={
                "type": "object",
                "required": ["left", "right"],
                "properties": {"left": {"type": "string"}, "right": {"type": "string"}, "family": {"type": "string"}},
            },
            output_schema={"type": "object"},
            side_effect_level=SideEffectLevel.none,
            availability=Availability.available,
            implementation_type=ImplementationType.python_entrypoint,
            required_dependencies=["optional pymatgen and/or ASE; core curated family grammar"],
            preconditions=["Two material identity strings are supplied."],
            postconditions=["No source files are modified. Ambiguous material strings remain ambiguous unless evidence resolves them."],
            provenance_policy="Record both source strings, parser/interpreter evidence, confidence, and ambiguity.",
            permissions_required=[PermissionLevel.read_only],
            enabled_by_default=True,
            owner_repo="https://github.com/scattering/tascopilot-paper",
        ),
    ]


def run_tool(call) -> ToolResult:
    dispatch = {
        ("tas.data_reader", "read_bt7_data"): read_bt7_data,
        ("tas.file_arithmetic", "combine_files"): combine_files,
        ("tas.corrections", "apply_corrections"): apply_corrections,
        ("tas.material_identity", "parse_formula"): _parse_formula_tool,
        ("tas.material_identity", "parse_filename_composition"): _parse_filename_composition_tool,
        ("tas.material_identity", "compare_material_identity"): _compare_material_identity_tool,
    }
    handler = dispatch.get((call.tool_namespace, call.tool_name))
    if handler is None:
        return ToolResult(status=ToolResultStatus.validation_failed, tool_name=call.tool_name, tool_namespace=call.tool_namespace)
    return handler(call.arguments)


def _parse_formula_tool(arguments: dict) -> ToolResult:
    text = str(arguments.get("text") or "").strip()
    if not text:
        return ToolResult(
            status=ToolResultStatus.missing_input,
            output={"missing_fields": ["text"]},
            tool_name="parse_formula",
            tool_namespace="tas.material_identity",
        )
    result = parse_formula(text, preferred_backend=str(arguments.get("preferred_backend") or "auto"))
    status = ToolResultStatus.unavailable if result.parser_backend == "unavailable" else ToolResultStatus.success
    return ToolResult(
        status=status,
        output=result.model_dump(mode="json"),
        warnings=result.warnings,
        provenance=result.provenance,
        tool_name="parse_formula",
        tool_namespace="tas.material_identity",
        tool_version="0.1.0",
    )


def _parse_filename_composition_tool(arguments: dict) -> ToolResult:
    text = str(arguments.get("text") or "").strip()
    if not text:
        return ToolResult(
            status=ToolResultStatus.missing_input,
            output={"missing_fields": ["text"]},
            tool_name="parse_filename_composition",
            tool_namespace="tas.material_identity",
        )
    family = str(arguments.get("family") or "lu_femn_o3")
    parsed = parse_material_text(text, family=family)
    record = material_identity_record(text, family=family, source_field=str(arguments.get("source_field") or "text"))
    warnings = list(dict.fromkeys([*parsed.warnings, *record.warnings]))
    return ToolResult(
        status=ToolResultStatus.success,
        output={
            "parse_result": parsed.model_dump(mode="json"),
            "material_identity": record.model_dump(mode="json"),
        },
        warnings=warnings,
        provenance=[*parsed.provenance, *record.provenance],
        tool_name="parse_filename_composition",
        tool_namespace="tas.material_identity",
        tool_version="0.1.0",
    )


def _compare_material_identity_tool(arguments: dict) -> ToolResult:
    left = str(arguments.get("left") or "").strip()
    right = str(arguments.get("right") or "").strip()
    missing = [field for field, value in {"left": left, "right": right}.items() if not value]
    if missing:
        return ToolResult(
            status=ToolResultStatus.missing_input,
            output={"missing_fields": missing},
            tool_name="compare_material_identity",
            tool_namespace="tas.material_identity",
        )
    output = compare_material_identity(left, right, family=str(arguments.get("family") or "lu_femn_o3"))
    return ToolResult(
        status=ToolResultStatus.success,
        output=output,
        warnings=[
            warning
            for side in ("left", "right")
            for warning in output.get(side, {}).get("warnings", [])
        ],
        tool_name="compare_material_identity",
        tool_namespace="tas.material_identity",
        tool_version="0.1.0",
    )
