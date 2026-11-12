from __future__ import annotations

from tas_copilot_contracts import ToolResult


def validate_tool_output_has_hashes(result: ToolResult) -> tuple[bool, str | None]:
    if not result.input_hash or not result.output_hash:
        return False, "tool result must include input_hash and output_hash"
    return True, None

