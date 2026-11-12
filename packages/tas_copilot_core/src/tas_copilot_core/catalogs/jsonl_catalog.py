from __future__ import annotations

import json
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonlCatalog(Generic[ModelT]):
    def __init__(self, path: str | Path, model: type[ModelT]) -> None:
        self.path = Path(path)
        self.model = model

    def load(self) -> list[ModelT]:
        if not self.path.exists():
            return []
        records: list[ModelT] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(self.model.model_validate(json.loads(line)))
        return records

    def save(self, records: Iterable[ModelT]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")

