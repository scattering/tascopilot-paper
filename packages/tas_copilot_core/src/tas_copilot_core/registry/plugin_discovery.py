from __future__ import annotations

import contextlib
import io
import importlib
import os
from importlib.metadata import entry_points
from typing import Protocol

from tas_copilot_contracts import ToolCall, ToolResult, ToolSpec


class PythonToolPlugin(Protocol):
    def get_tool_specs(self) -> list[ToolSpec]: ...

    def run_tool(self, call: ToolCall) -> ToolResult: ...


def discover_entrypoint_plugins(group: str = "tas_copilot.tools") -> list[PythonToolPlugin]:
    discovered = []
    for module_name in _plugin_modules_from_env():
        try:
            discovered.append(_load_module(module_name))
        except Exception:
            continue
    eps = entry_points()
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:
        selected = eps.get(group, [])
    for entry_point in selected:
        try:
            plugin = _load_entry_point(entry_point)
            discovered.append(plugin() if isinstance(plugin, type) else plugin)
        except Exception:
            continue
    return discovered


def _plugin_modules_from_env() -> list[str]:
    raw = os.environ.get("TAS_COPILOT_PLUGIN_MODULES", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _show_plugin_import_output() -> bool:
    raw = os.environ.get("TAS_COPILOT_SHOW_PLUGIN_IMPORT_OUTPUT", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_module(module_name: str):
    if _show_plugin_import_output():
        return importlib.import_module(module_name)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return importlib.import_module(module_name)


def _load_entry_point(entry_point):
    if _show_plugin_import_output():
        return entry_point.load()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return entry_point.load()
