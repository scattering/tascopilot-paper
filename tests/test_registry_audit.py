from pathlib import Path

from tas_copilot_contracts import (
    Availability,
    ImplementationType,
    PermissionLevel,
    SideEffectLevel,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry import PermissionPolicy, ToolRegistry, ToolRunner, load_tool_manifest


def _spec(name="ok", namespace="tas.test", permissions=None, availability=Availability.available):
    return ToolSpec(
        name=name,
        namespace=namespace,
        version="0.1",
        description="test",
        input_schema={},
        output_schema={},
        side_effect_level=SideEffectLevel.none,
        availability=availability,
        implementation_type=ImplementationType.stub,
        provenance_policy="record",
        permissions_required=permissions or [PermissionLevel.read_only],
    )


def test_manifest_loading():
    specs = load_tool_manifest(CoreConfig().tools_manifest_path)
    assert any(spec.namespace == "tas.resolution" for spec in specs)
    assert any(spec.namespace == "tas.scan_grouping" and spec.name == "group_bt7_scans" for spec in specs)


def test_registry_allowed_read_only_and_audit(tmp_path):
    registry = ToolRegistry()

    def handler(call):
        return ToolResult(status=ToolResultStatus.success, output={"ok": True}, tool_name=call.tool_name, tool_namespace=call.tool_namespace)

    registry.register(_spec(), handler)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path))
    result = runner.run(ToolCall(tool_name="ok", tool_namespace="tas.test"))
    assert result.status == ToolResultStatus.success
    assert result.audit_id
    assert (Path(tmp_path) / "audit.jsonl").exists()
    assert result.input_hash and result.output_hash


def test_unavailable_tool_behavior(tmp_path):
    registry = ToolRegistry()
    registry.register(_spec(availability=Availability.unavailable))
    result = ToolRunner(registry, JsonlAuditLogger(tmp_path)).run(ToolCall(tool_name="ok", tool_namespace="tas.test"))
    assert result.status == ToolResultStatus.unavailable


def test_blocked_live_control_and_disabled(tmp_path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="move",
            namespace="tas.instrument",
            version="0.1",
            description="bad",
            input_schema={},
            output_schema={},
            side_effect_level=SideEffectLevel.live_control,
            availability=Availability.available,
            implementation_type=ImplementationType.stub,
            provenance_policy="record",
            permissions_required=[PermissionLevel.live_control],
            enabled_by_default=False,
        )
    )
    result = ToolRunner(registry, JsonlAuditLogger(tmp_path)).run(ToolCall(tool_name="move", tool_namespace="tas.instrument"))
    assert result.status == ToolResultStatus.blocked


def test_simulation_allowed_by_policy():
    spec = _spec(permissions=[PermissionLevel.simulation_only])
    allowed, reason = PermissionPolicy().check(spec, ToolCall(tool_name=spec.name, tool_namespace=spec.namespace))
    assert allowed, reason


def test_writes_local_files_requires_lineage():
    spec = _spec(permissions=[PermissionLevel.writes_local_files])
    policy = PermissionPolicy()

    allowed, reason = policy.check(spec, ToolCall(tool_name=spec.name, tool_namespace=spec.namespace, arguments={"output_path": "out.dat"}))
    assert not allowed
    assert "lineage" in reason

    allowed, reason = policy.check(
        spec,
        ToolCall(
            tool_name=spec.name,
            tool_namespace=spec.namespace,
            arguments={"output_path": "out.dat", "input_datafile_ids": ["scan-1"]},
        ),
    )
    assert allowed, reason


def test_live_control_argument_blocked(tmp_path):
    registry = ToolRegistry()
    registry.register(_spec())
    result = ToolRunner(registry, JsonlAuditLogger(tmp_path)).run(ToolCall(tool_name="ok", tool_namespace="tas.test", arguments={"move_motor": "A3"}))
    assert result.status == ToolResultStatus.blocked
