from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from tas_copilot_contracts import (
    ProvenanceRecord,
    ToolError,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeInvocation,
    ToolRuntimeMode,
    ToolRuntimeProtocol,
    ToolRuntimeSpec,
)

from .tool_runner import stable_hash


class RuntimeToolRunner:
    """Run heavyweight external runtimes behind typed JSON boundaries."""

    def run_json(
        self,
        runtime: ToolRuntimeSpec,
        invocation: ToolRuntimeInvocation,
        *,
        tool_name: str,
        tool_namespace: str,
        tool_version: str | None = None,
    ) -> ToolResult:
        input_hash = stable_hash(invocation.model_dump(mode="json"))
        if runtime.mode == ToolRuntimeMode.unavailable:
            return _unavailable(runtime, invocation, tool_name, tool_namespace, tool_version, input_hash, "runtime is marked unavailable")
        if runtime.mode in {ToolRuntimeMode.container_once, ToolRuntimeMode.container_service, ToolRuntimeMode.remote_hpc_job}:
            return _unavailable(
                runtime,
                invocation,
                tool_name,
                tool_namespace,
                tool_version,
                input_hash,
                f"{runtime.mode.value} execution is represented but not enabled by this local runner",
            )
        if runtime.input_protocol != ToolRuntimeProtocol.stdin_stdout_json or runtime.output_protocol != ToolRuntimeProtocol.stdin_stdout_json:
            return _unavailable(
                runtime,
                invocation,
                tool_name,
                tool_namespace,
                tool_version,
                input_hash,
                "only stdin_stdout_json is implemented in the first runtime runner slice",
            )
        argv = _resolved_command(runtime, invocation)
        if not argv:
            return _unavailable(runtime, invocation, tool_name, tool_namespace, tool_version, input_hash, "runtime command is empty")
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in runtime.environment.items()})
        env.update({str(k): str(v) for k, v in invocation.environment.items()})
        cwd = Path(runtime.working_directory).expanduser() if runtime.working_directory else None
        try:
            completed = subprocess.run(
                argv,
                input=json.dumps(invocation.payload, sort_keys=True),
                capture_output=True,
                text=True,
                timeout=runtime.timeout_seconds,
                check=False,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
        except FileNotFoundError as exc:
            return _unavailable(runtime, invocation, tool_name, tool_namespace, tool_version, input_hash, f"runtime executable not found: {exc}")
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status=ToolResultStatus.error,
                output=None,
                errors=[ToolError(code="runtime_timeout", message=f"runtime exceeded {runtime.timeout_seconds}s", details={"cmd": argv})],
                tool_name=tool_name,
                tool_namespace=tool_namespace,
                tool_version=tool_version,
                input_hash=input_hash,
                runtime_environment=_runtime_environment(runtime, argv, completed=None),
                provenance=_runtime_provenance(runtime, invocation, "runtime_timeout"),
            )
        if completed.returncode != 0:
            return ToolResult(
                status=ToolResultStatus.error,
                output={"stdout": completed.stdout[-4000:] if completed.stdout else ""},
                errors=[
                    ToolError(
                        code="runtime_nonzero_exit",
                        message=f"runtime exited with code {completed.returncode}",
                        details={"stderr_tail": completed.stderr[-4000:] if completed.stderr else "", "cmd": argv},
                    )
                ],
                tool_name=tool_name,
                tool_namespace=tool_namespace,
                tool_version=tool_version,
                input_hash=input_hash,
                runtime_environment=_runtime_environment(runtime, argv, completed=completed),
                provenance=_runtime_provenance(runtime, invocation, "runtime_nonzero_exit"),
            )
        try:
            output: Any = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult(
                status=ToolResultStatus.validation_failed,
                output={"stdout": completed.stdout[-4000:] if completed.stdout else ""},
                errors=[ToolError(code="runtime_output_not_json", message=str(exc), details={"stderr_tail": completed.stderr[-4000:] if completed.stderr else ""})],
                tool_name=tool_name,
                tool_namespace=tool_namespace,
                tool_version=tool_version,
                input_hash=input_hash,
                runtime_environment=_runtime_environment(runtime, argv, completed=completed),
                provenance=_runtime_provenance(runtime, invocation, "runtime_output_validation"),
            )
        return ToolResult(
            status=ToolResultStatus.success,
            output=output,
            warnings=[],
            tool_name=tool_name,
            tool_namespace=tool_namespace,
            tool_version=tool_version,
            input_hash=input_hash,
            output_hash=stable_hash(output),
            runtime_environment=_runtime_environment(runtime, argv, completed=completed),
            provenance=_runtime_provenance(runtime, invocation, "runtime_json_execution"),
        )


def _resolved_command(runtime: ToolRuntimeSpec, invocation: ToolRuntimeInvocation) -> list[str]:
    python_exe = runtime.python_executable
    if runtime.mode == ToolRuntimeMode.conda_env and not python_exe and runtime.conda_env_path:
        python_exe = str(Path(runtime.conda_env_path).expanduser() / "bin" / "python")
    python_exe = python_exe or sys.executable
    values = {
        "{python}": str(Path(python_exe).expanduser()),
        "{output_dir}": str(Path(invocation.output_dir).expanduser()) if invocation.output_dir else "",
    }
    argv = [values.get(part, part) for part in runtime.command]
    argv.extend(invocation.extra_arguments)
    return [str(Path(item).expanduser()) if item.startswith("~/") else item for item in argv if item != ""]


def _unavailable(
    runtime: ToolRuntimeSpec,
    invocation: ToolRuntimeInvocation,
    tool_name: str,
    tool_namespace: str,
    tool_version: str | None,
    input_hash: str,
    message: str,
) -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.unavailable,
        output=None,
        errors=[ToolError(code="runtime_unavailable", message=message)],
        tool_name=tool_name,
        tool_namespace=tool_namespace,
        tool_version=tool_version,
        input_hash=input_hash,
        runtime_environment=_runtime_environment(runtime, runtime.command, completed=None),
        provenance=_runtime_provenance(runtime, invocation, "runtime_unavailable"),
    )


def _runtime_environment(runtime: ToolRuntimeSpec, argv: list[str], completed: subprocess.CompletedProcess[str] | None) -> dict[str, Any]:
    return {
        "runtime_id": runtime.runtime_id,
        "runtime_mode": runtime.mode.value,
        "command": argv,
        "container_image": runtime.container_image,
        "container_digest": runtime.container_digest,
        "apptainer_image": runtime.apptainer_image,
        "conda_env_name": runtime.conda_env_name,
        "conda_env_path": runtime.conda_env_path,
        "python_executable": runtime.python_executable,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "returncode": completed.returncode if completed is not None else None,
    }


def _runtime_provenance(runtime: ToolRuntimeSpec, invocation: ToolRuntimeInvocation, method: str) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            source_type="tool_runtime",
            source_id=runtime.runtime_id,
            source_uri=runtime.container_image or runtime.apptainer_image or runtime.conda_env_path or runtime.python_executable,
            method=method,
            metadata={
                "mode": runtime.mode.value,
                "invocation_id": invocation.invocation_id,
                "image_digest": runtime.container_digest,
                "read_only": runtime.read_only,
                "network_access": runtime.network_access,
            },
        )
    ]
