from __future__ import annotations

from tas_copilot_contracts import PermissionLevel, ToolCall, ToolSpec


class PermissionPolicy:
    def __init__(self, allow_human_approved: bool = False) -> None:
        self.allow_human_approved = allow_human_approved

    def check(self, spec: ToolSpec, call: ToolCall) -> tuple[bool, str | None]:
        if not spec.enabled_by_default:
            return False, "tool is disabled by default"
        required = set(spec.permissions_required)
        if PermissionLevel.disabled in required:
            return False, "tool has disabled permission"
        if PermissionLevel.live_control in required:
            return False, "live instrument control is outside the MVP"
        if PermissionLevel.requires_human_approval in required:
            if not (self.allow_human_approved and call.approval_token):
                return False, "human approval token required"
        if PermissionLevel.writes_local_files in required and not _has_local_lineage(call.arguments):
            return False, "writes_local_files tools require local derived artifact output and input lineage"
        return True, None


def _has_local_lineage(arguments: dict) -> bool:
    output_present = bool(
        arguments.get("output_path")
        or arguments.get("derived_output_path")
        or arguments.get("derived_artifact_id")
        or arguments.get("derived_artifacts")
        or arguments.get("in_memory") is True
    )
    lineage_present = bool(
        arguments.get("input_datafile_ids")
        or arguments.get("input_files")
        or arguments.get("source_datafile_ids")
        or _lineage_ids(arguments.get("lineage"))
    )
    return output_present and lineage_present


def _lineage_ids(lineage) -> bool:
    if not isinstance(lineage, dict):
        return False
    return bool(lineage.get("input_ids") or lineage.get("input_datafile_ids") or lineage.get("input_files"))
