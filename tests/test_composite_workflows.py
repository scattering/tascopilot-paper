from tas_copilot_contracts import ToolCall, ToolResult, ToolResultStatus
from tas_copilot_core.registry.tool_inventory import build_tool_inventory
from tas_copilot_core.composite_tools import run_tool
from tas_copilot_core.workflows.composite import apply_corrections, combine_files, read_bt7_data


class _FakeRunner:
    def __init__(self, status=ToolResultStatus.success):
        self.status = status
        self.calls = []

    def run(self, call):
        self.calls.append((call.tool_namespace, call.tool_name, call.arguments))
        return ToolResult(
            status=self.status,
            output={"called": f"{call.tool_namespace}.{call.tool_name}"},
            audit_id=f"audit-{len(self.calls)}",
            tool_name=call.tool_name,
            tool_namespace=call.tool_namespace,
        )


def test_read_bt7_data_composes_reader_tools():
    runner = _FakeRunner()
    result = read_bt7_data({"path": "/tmp/file.bt7"}, runner=runner)

    assert result.status == ToolResultStatus.success
    assert [call[1] for call in runner.calls] == ["classify_tas_format", "read_tas_metadata", "read_bt7_file"]
    assert result.output["tool_audit_ids"]["format"] == "audit-1"


def test_combine_files_dispatches_explicit_operation():
    runner = _FakeRunner()
    result = combine_files({"operation": "add_scans", "input_files": ["a.bt7", "b.bt7"]}, runner=runner)

    assert result.status == ToolResultStatus.success
    assert runner.calls[0][0:2] == ("tas.file_arithmetic", "add_scans")


def test_apply_corrections_runs_supported_steps(dag_runner):
    runner = dag_runner
    result = apply_corrections(
        {
            "steps": [
                {"tool": "normalize_to_monitor", "arguments": {"mode": "monitor"}},
                {"tool": "apply_detailed_balance", "arguments": {"temperature": {"value": 10, "unit": "K"}}},
            ],
            "input_files": ["scan.dat"],
            "scan": {"marker": "raw"},
        },
        runner=runner,
    )

    assert result.status == ToolResultStatus.success
    assert [call.tool_name for call in runner.calls] == ["normalize_to_monitor", "apply_detailed_balance"]
    assert runner.calls[1].arguments["scan"]["marker"] == 1
    assert result.output["scan"]["marker"] == 2


def test_composites_validate_required_semantics():
    assert read_bt7_data({}).status == ToolResultStatus.missing_input
    assert combine_files({"operation": "undefined"}).status == ToolResultStatus.missing_input
    assert apply_corrections({}).status == ToolResultStatus.missing_input


def test_core_composite_plugin_shadows_manifest_stubs(monkeypatch):
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tas_copilot_core.composite_tools")
    data = build_tool_inventory().as_dict()

    effective = {
        (tool["namespace"], tool["name"]): tool
        for tool in data["tools"]
        if tool["effective"]
    }
    assert effective[("tas.data_reader", "read_bt7_data")]["status"] == "available"
    assert effective[("tas.file_arithmetic", "combine_files")]["status"] == "available"
    assert effective[("tas.corrections", "apply_corrections")]["status"] == "available"
    assert effective[("tas.material_identity", "parse_formula")]["status"] == "available"
    assert effective[("tas.material_identity", "parse_filename_composition")]["status"] == "available"


def test_material_identity_tools_run_through_registry_plugin():
    parsed = run_tool(
        ToolCall(
            tool_namespace="tas.material_identity",
            tool_name="parse_formula",
            arguments={"text": "SrTiO3"},
        )
    )
    assert parsed.status == ToolResultStatus.success
    assert parsed.output["element_amounts"]["Sr"] == 1.0
    assert parsed.tool_namespace == "tas.material_identity"

    filename = run_tool(
        ToolCall(
            tool_namespace="tas.material_identity",
            tool_name="parse_filename_composition",
            arguments={"text": "LuFe75Mn25O3_Correlations102248.bt7", "family": "lu_femn_o3"},
        )
    )
    assert filename.status == ToolResultStatus.success
    assert filename.output["material_identity"]["x_variables"]["x_Mn"] == 0.25

    ambiguous = run_tool(
        ToolCall(
            tool_namespace="tas.material_identity",
            tool_name="parse_filename_composition",
            arguments={"text": "LuMnFeO3_29pt5_TdepWarm102246.bt7", "family": "lu_femn_o3"},
        )
    )
    assert ambiguous.status == ToolResultStatus.success
    assert ambiguous.output["material_identity"]["ambiguity_reason"]


def test_material_identity_compare_tool_preserves_matches_and_missing_inputs():
    compare = run_tool(
        ToolCall(
            tool_namespace="tas.material_identity",
            tool_name="compare_material_identity",
            arguments={"left": "LuFe0.6Mn0.4O3", "right": "LuFe60Mn40O3"},
        )
    )
    assert compare.status == ToolResultStatus.success
    assert compare.output["status"] == "match"

    missing = run_tool(
        ToolCall(
            tool_namespace="tas.material_identity",
            tool_name="compare_material_identity",
            arguments={"left": "SrTiO3"},
        )
    )
    assert missing.status == ToolResultStatus.missing_input
    assert missing.output["missing_fields"] == ["right"]
