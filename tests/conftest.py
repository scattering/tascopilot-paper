from copy import deepcopy

import pytest

from tas_copilot_contracts import (
    Availability, ImplementationType, PermissionLevel, SideEffectLevel,
    ToolResult, ToolResultStatus, ToolSpec,
)
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.reduction_pipeline import SUPPORTED_STEPS
from tas_copilot_core.registry import ToolRegistry, ToolRunner


@pytest.fixture
def dag_runner(tmp_path):
    """Real registry/permission/audit boundary with synthetic marker outputs."""
    calls = []

    def handler(call):
        calls.append(deepcopy(call))
        return ToolResult(
            status=ToolResultStatus.success, output={"scan": {"marker": len(calls)}},
            tool_namespace=call.tool_namespace, tool_name=call.tool_name, tool_version="fixture-1",
        )

    registry = ToolRegistry()
    for namespace, name in SUPPORTED_STEPS.values():
        registry.register(ToolSpec(
            namespace=namespace, name=name, version="fixture-1", description="Synthetic DAG fixture",
            input_schema={}, output_schema={}, side_effect_level=SideEffectLevel.local_files,
            availability=Availability.available, implementation_type=ImplementationType.python_entrypoint,
            provenance_policy="synthetic", permissions_required=[PermissionLevel.writes_local_files],
        ), handler)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    runner.calls = calls
    return runner
