from __future__ import annotations

from pathlib import Path

import yaml
from tas_copilot_contracts import ToolSpec


def load_tool_manifest(path: str | Path) -> list[ToolSpec]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    tools = raw.get("tools", raw if isinstance(raw, list) else [])
    return [ToolSpec.model_validate(tool) for tool in tools]

