from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tas_copilot_contracts import ToolCall, ToolResult, ToolSpec


ToolHandler = Callable[[ToolCall], ToolResult]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[tuple[str, str], RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler | None = None) -> None:
        self._tools[(spec.namespace, spec.name)] = RegisteredTool(spec=spec, handler=handler)

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, namespace: str, name: str) -> RegisteredTool:
        return self._tools[(namespace, name)]

    def list_specs(self) -> list[ToolSpec]:
        return [registered.spec for registered in self._tools.values()]

