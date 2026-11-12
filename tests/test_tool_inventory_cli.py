import json
import sys

from tas_copilot_contracts import Availability, ImplementationType, PermissionLevel, SideEffectLevel, ToolResult, ToolResultStatus, ToolSpec
from tas_copilot_core.cli import main
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_inventory import ToolInventory, InventoryRecord, build_tool_inventory, doctor_report
import tas_copilot_core.registry.tool_inventory as tool_inventory_module
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.testing import assert_tool_result_contract, assert_valid_plugin, assert_valid_tool_spec
from tas_copilot_core.workflows.inspect_bt7_directory import inspect_bt7_directory


def _spec(name="ok", namespace="tas.test", availability=Availability.available, enabled=True):
    return ToolSpec(
        name=name,
        namespace=namespace,
        version="0.1",
        description="fixture",
        input_schema={},
        output_schema={},
        side_effect_level=SideEffectLevel.none,
        availability=availability,
        implementation_type=ImplementationType.python_entrypoint,
        provenance_policy="record",
        permissions_required=[PermissionLevel.read_only],
        enabled_by_default=enabled,
    )


def test_tool_inventory_loads_manifest_and_reports_disabled():
    inventory = build_tool_inventory(CoreConfig())
    data = inventory.as_dict()

    assert any(tool["namespace"] == "tas.instrument_vm" and tool["status"] == "disabled" for tool in data["tools"])
    assert any(tool["namespace"] == "tas.data_reader" and tool["name"] == "scan_tas_directory" for tool in data["tools"])


def test_tool_inventory_clean_manifest_only_mode(monkeypatch):
    monkeypatch.setattr(tool_inventory_module, "discover_entrypoint_plugins", lambda: [])
    inventory = build_tool_inventory(CoreConfig())
    data = inventory.as_dict()

    assert data["summary"]["available"] == 0
    assert data["summary"]["unavailable"] > 0
    disabled_tools = {(tool["namespace"], tool["name"]) for tool in data["tools"] if tool["status"] == "disabled"}
    assert ("tas.instrument_vm", "simulate_control") in disabled_tools
    assert ("tas.psd_calibration", "channel_to_detector_angle") in disabled_tools
    assert ("tas.psd_calibration", "detector_angle_to_q") in disabled_tools
    assert data["summary"]["duplicates"] == 0


def test_inventory_duplicate_detection_and_doctor_report():
    inventory = ToolInventory(
        records=[
            InventoryRecord(spec=_spec(), source="manifest"),
            InventoryRecord(spec=_spec(), source="plugin:fixture", plugin_name="fixture", handler_available=True),
            InventoryRecord(spec=_spec("off", availability=Availability.disabled), source="manifest"),
        ]
    )

    data = inventory.as_dict()
    assert data["summary"]["duplicates"] == 0
    assert data["summary"]["shadowed_manifest_stubs"] == 1
    ok_tools = [tool for tool in data["tools"] if tool["name"] == "ok"]
    assert [tool for tool in ok_tools if tool["effective"]][0]["source"] == "plugin:fixture"
    report = doctor_report(inventory)
    assert "Manifest stubs shadowed by installed plugins" in report
    assert "disabled: tas.test.off" in report


def test_cli_tools_list_json_and_doctor(capsys):
    assert main(["tools", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert "tools" in listed

    assert main(["tools", "doctor"]) == 0
    report = capsys.readouterr().out
    assert "TASCopilot tool doctor" in report
    assert "What you can do today" in report
    assert "workflow copilot-ask --list-intents" in report
    assert "Runnable workflow starts today" in report
    assert "review labels backfill" in report

    assert main(["tools", "doctor", "--runtime-profile", "configs/runtime_profiles/nist_cpu.example.yaml"]) == 0
    report = capsys.readouterr().out
    assert "Runtime profile warnings" in report
    assert "example runtime profile selected" in report


def test_cli_without_args_prints_front_door_menu(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "TASCopilot" in output
    assert "workflow copilot-ask --list-intents" in output
    assert "Safety boundary" in output


def test_cli_examples_and_inventory_commands(capsys):
    assert main(["examples"]) == 0
    examples = capsys.readouterr().out
    assert "Bounded TASCopilot situations" in examples
    assert "experiment_review" in examples

    assert main(["inventory", "--json"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["workflow_count"] >= 1
    assert any(item["intent"] == "review_label_backfill" for item in inventory["workflows"])


def test_cli_initial_spine_commands_are_pinned(capsys):
    assert main(["analyze-dataset", "dataset-1"]) == 0
    analyzed = json.loads(capsys.readouterr().out)
    assert analyzed["identifier"] == "dataset-1"
    assert "retrieve-similar-scans" in analyzed["available_next_actions"]

    assert main(["extract-setup", "fixed final energy Ef=14.7 meV collimation 40-40-40-80 PG filter"]) == 0
    setup = json.loads(capsys.readouterr().out)
    assert setup["fixed_final_energy"] == "14.7"
    assert setup["filter"] == "PG filter"

    assert main(["combine-files", "--json", '{"operation":"undefined"}']) == 0
    combined = json.loads(capsys.readouterr().out)
    assert combined["status"] == ToolResultStatus.missing_input.value

    assert main(["apply-corrections", "--json", "{}"]) == 0
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["status"] == ToolResultStatus.missing_input.value

    assert main(["powder-order-parameter", "--json", "{}"]) == 0
    powder = json.loads(capsys.readouterr().out)
    assert powder["tool_namespace"] == "tas.andie"
    assert powder["status"] in {ToolResultStatus.unavailable.value, ToolResultStatus.missing_input.value}


def test_cli_plugin_backed_verbs_are_pinned(capsys):
    commands = {
        "check-spurions": ("tas.spurion", "check_spurions"),
        "check-resolution": ("tas.resolution", "calculate_resolution"),
        "calculate-angles": ("tas.device_angles", "calculate_angles"),
        "calculate-ub": ("tas.ubmatrix", "calculate_ub"),
        "classify-tas-format": ("tas.data_reader", "classify_tas_format"),
        "scan-tas-directory": ("tas.data_reader", "scan_tas_directory"),
        "read-tas-metadata": ("tas.data_reader", "read_tas_metadata"),
        "read-bt7-file": ("tas.data_reader", "read_bt7_file"),
    }
    safe_statuses = {
        ToolResultStatus.success.value,
        ToolResultStatus.unavailable.value,
        ToolResultStatus.missing_input.value,
        ToolResultStatus.validation_failed.value,
        ToolResultStatus.blocked.value,
        ToolResultStatus.error.value,
    }

    for command, (namespace, tool_name) in commands.items():
        assert main([command, "--json", "{}"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["tool_namespace"] == namespace
        assert result["tool_name"] == tool_name
        assert result["status"] in safe_statuses


def test_plugin_discovery_suppresses_import_output_by_default(monkeypatch, tmp_path, capsys):
    module_name = "noisy_plugin_for_inventory_test"
    (tmp_path / f"{module_name}.py").write_text(
        "\n".join(
            [
                "import sys",
                "print('noisy plugin stdout')",
                "print('noisy plugin stderr', file=sys.stderr)",
                "def get_tool_specs():",
                "    return []",
                "def run_tool(call):",
                "    raise NotImplementedError",
            ]
        )
    )
    sys.modules.pop(module_name, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", module_name)
    monkeypatch.delenv("TAS_COPILOT_SHOW_PLUGIN_IMPORT_OUTPUT", raising=False)

    plugins = discover_entrypoint_plugins()

    assert any(getattr(plugin, "__name__", "") == module_name for plugin in plugins)
    captured = capsys.readouterr()
    assert "noisy plugin" not in captured.out
    assert "noisy plugin" not in captured.err


def test_cli_workflow_aliases_for_compare_and_scan_geometry(capsys, tmp_path):
    assert main(["workflow", "compare-paper-to-data", "--paper-id", "paper-missing"]) == 0
    compare = json.loads(capsys.readouterr().out)
    assert compare["status"] == ToolResultStatus.missing_input.value

    missing_db = tmp_path / "missing.sqlite"
    assert main(["workflow", "check-geometry", "--scan-id", "scan-missing", "--db", str(missing_db)]) == 0
    geometry = json.loads(capsys.readouterr().out)
    assert geometry["status"] == ToolResultStatus.missing_input.value
    assert "scan_id.catalog_missing" in geometry["missing_fields"]


def test_cli_workflow_aliases_for_reduce_and_similar_scans(capsys):
    assert main(["workflow", "reduce-scan-set", "--json", '{"steps":[{"tool":"add_scans"}]}']) == 0
    reduce = json.loads(capsys.readouterr().out)
    assert reduce["status"] == ToolResultStatus.missing_input.value
    assert any(item["code"] == "missing_input_lineage" for item in reduce["validation_failures"])

    assert main(["workflow", "retrieve-similar-scans", "scan-001", "--limit", "1"]) == 0
    similar = json.loads(capsys.readouterr().out)
    assert len(similar) == 1
    assert "similarity_score" in similar[0]


class _Plugin:
    @staticmethod
    def get_tool_specs():
        return [_spec()]

    @staticmethod
    def run_tool(call):
        return ToolResult(status=ToolResultStatus.validation_failed, tool_name=call.tool_name, tool_namespace=call.tool_namespace)


def test_plugin_contract_helpers():
    spec = assert_valid_tool_spec(_spec())
    assert spec.name == "ok"
    result = assert_tool_result_contract(ToolResult(status=ToolResultStatus.success, tool_name="ok", tool_namespace="tas.test"))
    assert result.status == ToolResultStatus.success
    assert_valid_plugin(_Plugin)


class _FakeRunner:
    def __init__(self, status=ToolResultStatus.success):
        self.status = status
        self.calls = []

    def run(self, call):
        self.calls.append((call.tool_namespace, call.tool_name, call.arguments))
        output = {}
        if call.tool_name == "scan_tas_directory" and self.status == ToolResultStatus.success:
            output = {"records": [{"path": "/tmp/example.bt7"}], "format_counts": {"ncnr_bt7_ice_nice": 1}, "instrument_counts": {"BT-7": 1}}
        if call.tool_name == "group_bt7_scans" and self.status == ToolResultStatus.success:
            output = {"group_count": 1, "groups": [{"record_paths": ["/tmp/example.bt7"]}]}
        return ToolResult(status=self.status, output=output, tool_name=call.tool_name, tool_namespace=call.tool_namespace)


def test_inspect_bt7_directory_workflow_composes_registry_tools(tmp_path):
    runner = _FakeRunner()
    result = inspect_bt7_directory({"path": "/tmp/bt7", "max_metadata_files": 1}, runner=runner, audit_logger=JsonlAuditLogger(tmp_path))

    assert result["status"] == "success"
    assert result["discovered_file_count"] == 1
    assert [call[1] for call in runner.calls] == [
        "scan_tas_directory",
        "classify_tas_format",
        "read_tas_metadata",
        "group_bt7_scans",
    ]
    assert result["workflow_audit_id"]


def test_inspect_bt7_directory_unavailable_when_plugins_missing(tmp_path):
    runner = _FakeRunner(status=ToolResultStatus.unavailable)
    result = inspect_bt7_directory({"path": "/tmp/bt7"}, runner=runner, audit_logger=JsonlAuditLogger(tmp_path))

    assert result["status"] == "unavailable"
    assert result["discovered_file_count"] == 0
