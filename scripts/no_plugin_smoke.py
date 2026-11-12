from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tas_copilot_core.cli import main


def run() -> int:
    """Exercise core fail-closed behavior without requiring external plugins."""

    with tempfile.TemporaryDirectory(prefix="tascopilot-no-plugin-") as tmp:
        missing_db = Path(tmp) / "missing.sqlite"
        commands = [
            ["tools", "doctor"],
            ["tools", "list", "--json"],
            ["workflow", "check-geometry", "--scan-id", "missing", "--db", str(missing_db)],
            ["workflow", "compare-paper-to-data", "--paper-id", "paper-1", "--json", json.dumps({"db_path": str(missing_db)})],
        ]
        for command in commands:
            rc = main(command)
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
