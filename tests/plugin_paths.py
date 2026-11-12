"""Explicit, opt-in checkouts for external scientific integration tests."""

import os
from pathlib import Path


def plugin_path(name: str) -> Path:
    root = os.environ.get("TAS_COPILOT_TEST_PLUGIN_ROOT")
    base = Path(root).expanduser() if root else Path(__file__).resolve().parents[1] / "local_runs" / "test_plugins"
    return base / name
