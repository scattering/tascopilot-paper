from pathlib import Path

from plugin_paths import plugin_path
import importlib

import pytest

from tas_copilot_contracts import ToolCall, ToolResult, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.registry import ToolRegistry, ToolRunner
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.workflows.check_resolution import _scattering_plane_bz_summary, _scan_record_from_result, write_resolution_plotly_artifacts


RESCALCULATOR_PATH = (plugin_path("rescalculator"))


def _import_rescalculator_plugin(monkeypatch):
    monkeypatch.syspath_prepend(str(RESCALCULATOR_PATH))
    try:
        return importlib.import_module("rescalculator.copilot_plugin")
    except ModuleNotFoundError:
        return importlib.import_module("copilot_plugin")


def _skip_if_optional_dependency_missing(result):
    if result.status == ToolResultStatus.unavailable and any(error.code == "optional_dependency_missing" for error in result.errors):
        pytest.skip(result.errors[0].message)


def _arguments(backend="numpy"):
    return {
        "q_point": {"h": 1.0, "k": 0.0, "l": 0.0},
        "energy_transfer": {"value": {"value": 5.0, "unit": "meV"}},
        "instrument_config": {
            "fixed_energy_mode": "fixed_final_energy",
            "fixed_energy": {"value": 14.7, "unit": "meV"},
            "monochromator": "PG(002)",
            "analyzer": "PG(002)",
            "collimation": "40-40-40-80",
        },
        "lattice": {
            "a": 5.0,
            "b": 5.0,
            "c": 5.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "orient1": [[1, 0, 0]],
            "orient2": [[0, 1, 0]],
        },
        "backend": backend,
    }


def test_resolution_plotly_artifacts_from_matrix(tmp_path):
    result = ToolResult(
        status=ToolResultStatus.success,
        tool_namespace="tas.resolution",
        tool_name="calculate_resolution",
        output={
            "resolution": {
                "resolution_matrix": [
                    [8.0, 1.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
        },
        audit_id="audit-test",
        input_hash="input-test",
        output_hash="output-test",
    )

    artifacts = write_resolution_plotly_artifacts(result, tmp_path, _arguments())

    artifact_types = [artifact["artifact_type"] for artifact in artifacts]
    assert "resolution_plotly_json" in artifact_types
    assert "resolution_plotly_html" in artifact_types
    html_path = tmp_path / "resolution_ellipse_projections.html"
    json_path = tmp_path / "resolution_ellipse_projections.json"
    assert html_path.exists()
    assert json_path.exists()
    assert "Plotly.newPlot" in html_path.read_text(encoding="utf-8")
    text = json_path.read_text(encoding="utf-8")
    assert "Qx_Qy" in text
    assert "projection" in text
    assert "cross_section" in text
    assert "x_secondary_axis" in text
    assert "Qx units of [1 0 0]" in text
    html = html_path.read_text(encoding="utf-8")
    assert "xaxis2" in html
    assert "yaxis2" in html


def test_resolution_plotly_artifacts_center_scan_points(tmp_path):
    result = ToolResult(
        status=ToolResultStatus.success,
        tool_namespace="tas.resolution",
        tool_name="calculate_resolution",
        output={
            "resolution_scan": {
                "points": [
                    {
                        "point_index": 0,
                        "q_point": {"h": 1.0, "k": 0.0, "l": 0.0},
                        "energy_transfer_meV": 0.0,
                        "center": {"qx_inv_angstrom": 1.0, "qy_inv_angstrom": 0.0, "energy_meV": 0.0},
                        "resolution_matrix": [
                            [8.0, 1.0, 0.0, 0.0],
                            [1.0, 4.0, 0.0, 0.0],
                            [0.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    {
                        "point_index": 1,
                        "q_point": {"h": 1.1, "k": 0.0, "l": 0.0},
                        "energy_transfer_meV": 0.5,
                        "center": {"qx_inv_angstrom": 1.1, "qy_inv_angstrom": 0.0, "energy_meV": 0.5},
                        "resolution_matrix": [
                            [8.0, 1.0, 0.0, 0.0],
                            [1.0, 4.0, 0.0, 0.0],
                            [0.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                ]
            }
        },
    )

    write_resolution_plotly_artifacts(result, tmp_path)

    text = (tmp_path / "resolution_ellipse_projections.json").read_text(encoding="utf-8")
    assert '"point_count": 2' in text
    assert '"scan_path"' in text
    assert '"center"' in text


def test_scan_record_uses_reciprocal_inverse_angstrom_center():
    result = ToolResult(
        status=ToolResultStatus.success,
        tool_namespace="tas.resolution",
        tool_name="calculate_resolution",
        output={"resolution": {"resolution_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}},
    )

    record = _scan_record_from_result(_arguments(), result, point_index=0)

    assert record["center"]["qx_inv_angstrom"] > 0
    assert record["center"]["energy_meV"] == 5.0


def test_scattering_plane_bz_summary_handles_nonorthogonal_lattice():
    summary = _scattering_plane_bz_summary(
        {
            "q_point": {"h": 1, "k": 0, "l": 0},
            "lattice": {
                "a": 3.0,
                "b": 3.0,
                "c": 5.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 120.0,
                "orient1": [[1, 0, 0]],
                "orient2": [[0, 1, 0]],
            },
        }
    )

    assert summary["visualization_type"] == "pymatgen_scattering_plane_bz_section"
    assert len(summary["bz_section"]["x"]) >= 4
    assert summary["basis"]["normal_cartesian"]
    assert "hkl_basis" in summary["coordinate_modes"]


def test_scattering_plane_bz_summary_can_select_hkl_basis_for_nonorthogonal_lattice():
    summary = _scattering_plane_bz_summary(
        {
            "bz_coordinate_mode": "hkl_basis",
            "q_point": {"h": 1, "k": 0, "l": 0},
            "lattice": {
                "a": 3.0,
                "b": 3.0,
                "c": 5.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 120.0,
                "orient1": [[1, 0, 0]],
                "orient2": [[0, 1, 0]],
            },
        }
    )

    assert summary["selected_coordinate_mode"] == "hkl_basis"
    assert summary["coordinate_system"] == "non-orthogonal HKL scattering-plane basis"
    assert summary["q_point"]["plane_coordinates"] == pytest.approx([1.0, 0.0], abs=1e-10)
    assert summary["coordinate_modes"]["hkl_basis"]["warning"]


@pytest.mark.skipif(not RESCALCULATOR_PATH.exists(), reason="local rescalculator checkout is not available")
def test_local_rescalculator_plugin_registers_and_runs_resolution(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(RESCALCULATOR_PATH))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "copilot_plugin")
    assert any(plugin.__name__ in {"rescalculator.copilot_plugin", "copilot_plugin"} for plugin in discover_entrypoint_plugins())
    copilot_plugin = _import_rescalculator_plugin(monkeypatch)

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, copilot_plugin.run_tool)

    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    result = runner.run(
        ToolCall(
            tool_namespace="tas.resolution",
            tool_name="calculate_resolution",
            arguments=_arguments(),
            caller_workflow="test",
        )
    )

    _skip_if_optional_dependency_missing(result)
    assert result.status == ToolResultStatus.success
    assert result.audit_id
    assert result.output["backend_used"] == "numpy"
    assert result.output["resolution"]["resolution_matrix"]


@pytest.mark.skipif(not RESCALCULATOR_PATH.exists(), reason="local rescalculator checkout is not available")
def test_local_rescalculator_plugin_registers_and_runs_angles(tmp_path, monkeypatch):
    copilot_plugin = _import_rescalculator_plugin(monkeypatch)

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, copilot_plugin.run_tool)

    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    result = runner.run(
        ToolCall(
            tool_namespace="tas.device_angles",
            tool_name="calculate_angles",
            arguments=_arguments(),
            caller_workflow="test",
        )
    )

    _skip_if_optional_dependency_missing(result)
    assert result.status == ToolResultStatus.success
    assert [item["name"] for item in result.output["device_angles"]["angles"]] == ["M1", "M2", "S1", "S2", "A1", "A2"]


@pytest.mark.skipif(not RESCALCULATOR_PATH.exists(), reason="local rescalculator checkout is not available")
def test_local_rescalculator_plugin_registers_and_runs_ub(tmp_path, monkeypatch):
    copilot_plugin = _import_rescalculator_plugin(monkeypatch)

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, copilot_plugin.run_tool)

    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    result = runner.run(
        ToolCall(
            tool_namespace="tas.ubmatrix",
            tool_name="calculate_ub",
            arguments=_arguments(),
            caller_workflow="test",
        )
    )

    _skip_if_optional_dependency_missing(result)
    assert result.status == ToolResultStatus.success
    assert result.audit_id
    assert result.output["ub_matrix"]["matrix"]
    assert result.output["reciprocal_lattice"]["astar"] > 0
