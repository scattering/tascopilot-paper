from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_core.testing import assert_valid_plugin


@dataclass(frozen=True)
class PluginCase:
    name: str
    path: Path
    import_path: str
    python_path: Path | None = None


PLUGIN_CASES = [
    PluginCase("tas-data-readers", (plugin_path("tas-data-readers")), "tas_data_readers.copilot_plugin", (plugin_path("tas-data-readers") / "src")),
    PluginCase("rescalculator", (plugin_path("rescalculator")), "copilot_plugin", (plugin_path("rescalculator"))),
    PluginCase("spurion", (plugin_path("spurion")), "tas_spurion.copilot_plugin", (plugin_path("spurion"))),
    PluginCase("bt7-paper-mapping", (plugin_path("bt7-paper-mapping")), "bt7_paper_mapping.copilot_plugin", (plugin_path("bt7-paper-mapping") / "src")),
    PluginCase("tas-pyspinw-adapter", (plugin_path("tas-pyspinw-adapter")), "tas_pyspinw_adapter.copilot_plugin", (plugin_path("tas-pyspinw-adapter") / "src")),
    PluginCase("bt7grouping", (plugin_path("bt7grouping")), "bt7grouping.copilot_plugin", (plugin_path("bt7grouping"))),
    PluginCase("tasreduction", (plugin_path("tasreduction")), "tasreduction.copilot_plugin", (plugin_path("tasreduction"))),
    PluginCase("psdplanner", (plugin_path("psdplanner")), "tas_psd_planner.copilot_plugin", (plugin_path("psdplanner") / "src")),
]


@pytest.mark.parametrize("case", PLUGIN_CASES, ids=[case.name for case in PLUGIN_CASES])
def test_external_plugin_contracts_when_checkout_is_present(case, monkeypatch):
    if not case.path.exists():
        pytest.skip(f"local {case.name} checkout is not available")
    if case.python_path is not None:
        monkeypatch.syspath_prepend(str(case.python_path))

    plugin = importlib.import_module(case.import_path)
    specs = assert_valid_plugin(plugin)

    assert specs
    assert all(spec.namespace.startswith("tas.") for spec in specs)
