"""Disposable, content-checked local JSON cache for successful DAG node results."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from tas_copilot_contracts import ReductionNodeResult
from tas_copilot_core.registry.tool_runner import stable_hash


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ReductionCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    def load(self, key: str, node_id: str) -> ReductionNodeResult | None:
        try:
            entry = json.loads((self.directory / f"{key}.json").read_text(encoding="utf-8"))
            payload = entry["result"]
            if entry["schema_version"] != 1 or entry["result_hash"] != stable_hash(payload):
                return None
            result = ReductionNodeResult.model_validate(payload)
            if result.cache_key != key or result.node_id != node_id or result.status != "success":
                return None
            return result
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return None

    def save(self, result: ReductionNodeResult) -> None:
        if result.status != "success":
            return
        payload = result.model_dump(mode="json")
        write_json_atomic(self.directory / f"{result.cache_key}.json", {
            "schema_version": 1,
            "result_hash": stable_hash(payload),
            "result": payload,
        })
