import json
import sys

import pytest

from tas_copilot_contracts import (
    ToolResultStatus,
    ToolRuntimeInvocation,
    ToolRuntimeMode,
    ToolRuntimeSpec,
)
from tas_copilot_core.registry import RuntimeToolRunner


def test_runtime_spec_requires_explicit_command_for_conda_env():
    with pytest.raises(ValueError):
        ToolRuntimeSpec(runtime_id="bad", mode=ToolRuntimeMode.conda_env)


def test_runtime_runner_native_json_success():
    runtime = ToolRuntimeSpec(
        runtime_id="echo-native",
        mode=ToolRuntimeMode.native_python,
        command=[
            sys.executable,
            "-c",
            "import json,sys; payload=json.load(sys.stdin); print(json.dumps({'received': payload, 'runtime': 'native'}))",
        ],
    )
    result = RuntimeToolRunner().run_json(
        runtime,
        ToolRuntimeInvocation(runtime_id=runtime.runtime_id, payload={"x": 3}),
        tool_name="echo",
        tool_namespace="tas.runtime_test",
    )

    assert result.status == ToolResultStatus.success
    assert result.output == {"received": {"x": 3}, "runtime": "native"}
    assert result.runtime_environment["runtime_mode"] == "native_python"
    assert result.provenance[0].source_id == "echo-native"


def test_runtime_runner_conda_json_uses_explicit_python():
    runtime = ToolRuntimeSpec(
        runtime_id="echo-conda",
        mode=ToolRuntimeMode.conda_env,
        python_executable=sys.executable,
        command=[
            "{python}",
            "-c",
            "import json,sys; payload=json.load(sys.stdin); print(json.dumps({'ok': True, 'payload': payload}))",
        ],
    )
    result = RuntimeToolRunner().run_json(
        runtime,
        ToolRuntimeInvocation(runtime_id=runtime.runtime_id, payload={"sample": "SrTiO3"}),
        tool_name="echo",
        tool_namespace="tas.runtime_test",
    )

    assert result.status == ToolResultStatus.success
    assert result.output["payload"]["sample"] == "SrTiO3"
    assert result.runtime_environment["command"][0] == sys.executable


def test_runtime_runner_container_mode_fails_closed():
    runtime = ToolRuntimeSpec(
        runtime_id="gsasii-container",
        mode=ToolRuntimeMode.container_once,
        container_image="ghcr.io/scattering/tas-gsasii:example",
        container_digest="sha256:example",
    )
    result = RuntimeToolRunner().run_json(
        runtime,
        ToolRuntimeInvocation(runtime_id=runtime.runtime_id, payload={}),
        tool_name="structure_factor",
        tool_namespace="tas.gsasii",
    )

    assert result.status == ToolResultStatus.unavailable
    assert result.errors[0].code == "runtime_unavailable"
    assert result.runtime_environment["container_digest"] == "sha256:example"


def test_runtime_runner_rejects_non_json_stdout():
    runtime = ToolRuntimeSpec(
        runtime_id="bad-json",
        mode=ToolRuntimeMode.native_python,
        command=[sys.executable, "-c", "print('not json')"],
    )
    result = RuntimeToolRunner().run_json(
        runtime,
        ToolRuntimeInvocation(runtime_id=runtime.runtime_id, payload={}),
        tool_name="bad",
        tool_namespace="tas.runtime_test",
    )

    assert result.status == ToolResultStatus.validation_failed
    assert result.errors[0].code == "runtime_output_not_json"


def test_gsasii_runtime_config_loads():
    import yaml

    with open("configs/tool_runtimes/gsasii.yaml", "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    spec = ToolRuntimeSpec.model_validate(raw)

    assert spec.runtime_id == "gsasii-local-conda"
    assert spec.mode == ToolRuntimeMode.conda_env
    assert spec.read_only is True


@pytest.mark.parametrize(
    ("path", "runtime_id"),
    [
        ("configs/tool_runtimes/tasai_tacc.yaml", "tasai-tacc-remote-hpc"),
        ("configs/tool_runtimes/andie_tacc.yaml", "andie-tacc-remote-hpc"),
        ("configs/tool_runtimes/pyspinw_tacc.yaml", "pyspinw-tacc-remote-hpc"),
    ],
)
def test_adaptive_tacc_runtime_configs_fail_closed(path: str, runtime_id: str):
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    spec = ToolRuntimeSpec.model_validate(raw)
    result = RuntimeToolRunner().run_json(
        spec,
        ToolRuntimeInvocation(runtime_id=spec.runtime_id, payload={}),
        tool_name="adaptive",
        tool_namespace="tas.runtime_test",
    )

    assert spec.runtime_id == runtime_id
    assert spec.mode == ToolRuntimeMode.remote_hpc_job
    assert result.status == ToolResultStatus.unavailable
    assert "not enabled" in result.errors[0].message
