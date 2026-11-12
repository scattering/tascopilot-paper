"""Asynchronous, inference-only Slurm jobs with immutable JSON boundaries.

Scientific models live in external workers. ``prepare`` is local; submit/status/
cancel/fetch require explicit ``allow_remote=True``. Tests inject an argv runner.
This module never connects to NICE or contains an instrument-control interface.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence


class InferenceJobError(RuntimeError):
    pass


class RemoteExecutionDisabled(InferenceJobError):
    pass


class IdentityMismatch(InferenceJobError):
    pass


class UnknownSubmission(InferenceJobError):
    """Submission may have happened. Reconcile the scheduler; never resubmit."""


class TransportError(InferenceJobError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts or str(path) != value or "\x00" in value:
        raise ValueError(f"artifact path must be a normalized relative path: {value!r}")
    return value


def _names(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x.strip() for x in result) or len(set(result)) != len(result):
        raise ValueError("parameter_names must be nonempty, unique, ordered names")
    return result


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    size_bytes: int | None = None

    def __post_init__(self):
        _relative_path(self.path)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("artifact sha256 must contain 64 lowercase hexadecimal digits")
        if self.size_bytes is not None and (not isinstance(self.size_bytes, int) or self.size_bytes < 0):
            raise ValueError("size_bytes must be nonnegative")

    @classmethod
    def from_file(cls, path: str, source: str | Path) -> "ArtifactRef":
        source = Path(source)
        return cls(path, file_hash(source), source.stat().st_size)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(value["path"], value["sha256"], value.get("size_bytes"))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    def validate_file(self, root: Path) -> Path:
        root = root.resolve()
        path = root / self.path
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise IdentityMismatch(f"missing or escaping artifact: {self.path}")
        if self.size_bytes is not None and path.stat().st_size != self.size_bytes:
            raise IdentityMismatch(f"artifact size differs: {self.path}")
        if file_hash(path) != self.sha256:
            raise IdentityMismatch(f"artifact checksum differs: {self.path}")
        return path


@dataclass(frozen=True)
class InferenceJobRequest:
    payload: Mapping[str, Any]
    parameter_names: Sequence[str]
    artifacts: Sequence[ArtifactRef] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "parameter_names", _names(self.parameter_names))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if len({x.path for x in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate input artifact paths")
        # Detach caller-owned dictionaries and reject NaN/non-JSON values.
        object.__setattr__(self, "payload", json.loads(canonical_json(self.payload)))
        object.__setattr__(self, "provenance", json.loads(canonical_json(self.provenance)))

    def _content(self) -> dict[str, Any]:
        return {"schema_version": 1, "payload": self.payload,
                "parameter_names": list(self.parameter_names),
                "artifacts": [x.to_dict() for x in self.artifacts], "provenance": self.provenance}

    @property
    def input_hash(self) -> str:
        return content_hash(self._content())

    @property
    def request_id(self) -> str:
        return "inference-" + self.input_hash[:24]

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "request_id": self.request_id, "input_hash": self.input_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceJobRequest":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported inference request schema")
        result = cls(value["payload"], value["parameter_names"],
                     tuple(ArtifactRef.from_dict(x) for x in value.get("artifacts", [])), value.get("provenance", {}))
        if value.get("input_hash") != result.input_hash or value.get("request_id") != result.request_id:
            raise IdentityMismatch("request content differs from its declared identity")
        return result


@dataclass(frozen=True)
class CheckpointRef:
    input_hash: str
    parameter_names: Sequence[str]
    artifact: ArtifactRef
    runtime_fingerprint: str

    def validate(self, request: InferenceJobRequest, runtime_fingerprint: str) -> None:
        if (self.input_hash != request.input_hash or tuple(self.parameter_names) != request.parameter_names
                or self.runtime_fingerprint != runtime_fingerprint):
            raise IdentityMismatch("checkpoint data/model/parameter order/runtime identity differs")

    def to_dict(self) -> dict[str, Any]:
        return {"input_hash": self.input_hash, "parameter_names": list(self.parameter_names),
                "artifact": self.artifact.to_dict(), "runtime_fingerprint": self.runtime_fingerprint}


@dataclass(frozen=True)
class InferenceJobResult:
    request_id: str
    input_hash: str
    job_id: str
    parameter_names: Sequence[str]
    status: str
    posterior_artifacts: Sequence[ArtifactRef] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "parameter_names", _names(self.parameter_names))
        object.__setattr__(self, "posterior_artifacts", tuple(self.posterior_artifacts))
        if self.status not in {"completed", "failed", "unavailable"}:
            raise ValueError("invalid inference result status")
        if self.status == "completed" and not self.posterior_artifacts:
            raise ValueError("completed result requires posterior artifact references")
        if len({x.path for x in self.posterior_artifacts}) != len(self.posterior_artifacts):
            raise ValueError("duplicate posterior artifact paths")
        canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "request_id": self.request_id, "input_hash": self.input_hash,
                "job_id": self.job_id, "parameter_names": list(self.parameter_names), "status": self.status,
                "posterior_artifacts": [x.to_dict() for x in self.posterior_artifacts],
                "diagnostics": self.diagnostics, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceJobResult":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported inference result schema")
        return cls(value["request_id"], value["input_hash"], str(value["job_id"]), value["parameter_names"],
                   value["status"], tuple(ArtifactRef.from_dict(x) for x in value.get("posterior_artifacts", [])),
                   value.get("diagnostics", {}), value.get("provenance", {}))

    def validate(self, request: InferenceJobRequest, job_id: str, root: Path) -> None:
        if (self.request_id != request.request_id or self.input_hash != request.input_hash
                or self.job_id != job_id or tuple(self.parameter_names) != request.parameter_names):
            raise IdentityMismatch("stale or incompatible result identity/parameter order")
        for artifact in self.posterior_artifacts:
            artifact.validate_file(root)


@dataclass(frozen=True)
class HPCProfile:
    profile_id: str
    login_host: str
    remote_root: str
    partition: str
    account: str | None = None
    controlpath: str | None = None
    cpus_per_task: int = 8
    walltime: str = "02:00:00"
    python_executable: str | None = None
    runtime_version: str = "unverified"
    recorded_not_live_verified: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.profile_id):
            raise ValueError("invalid profile id")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", self.login_host):
            raise ValueError("invalid SSH login host")
        if (not re.fullmatch(r"/[A-Za-z0-9_./-]+", self.remote_root)
                or ".." in PurePosixPath(self.remote_root).parts):
            raise ValueError("remote_root must be an expanded absolute path")
        if self.cpus_per_task < 1 or not re.fullmatch(r"[0-9]+:[0-9]{2}:[0-9]{2}", self.walltime):
            raise ValueError("invalid CPU allocation or walltime")
        for value in (self.partition, self.account):
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_+-]+", value):
                raise ValueError("invalid scheduler account/partition")

    @classmethod
    def from_file(cls, path: str | Path, **overrides: Any) -> "HPCProfile":
        import yaml
        value = yaml.safe_load(Path(path).read_text())
        if value.get("scheduler") != "slurm":
            raise ValueError("only Slurm profiles are supported")
        config = value.get("inference") or {}
        fields = dict(profile_id=value["profile_id"], login_host=value["login_host"],
                      remote_root=value["remote_root"], partition=value["default_partition"],
                      account=value.get("default_account"), controlpath=value.get("controlpath"),
                      cpus_per_task=config.get("cpus_per_task", 8), walltime=config.get("walltime", "02:00:00"),
                      python_executable=config.get("python_executable"), runtime_version=config.get("runtime_version", "unverified"),
                      recorded_not_live_verified=value.get("recorded_not_live_verified", True))
        fields.update(overrides)
        return cls(**fields)

    @property
    def fingerprint(self) -> str:
        return content_hash(self.__dict__)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class InferenceJobBackend:
    """Durable Slurm lifecycle; runner(argv) returns CompletedProcess-like data.

    ``command`` in prepare is trusted adapter configuration, never request data.
    Supported placeholders: {request}, {output}, {input}, {checkpoint}, {python},
    {cpus}. Workers must emit result.json using InferenceJobResult's envelope.
    """

    def __init__(self, profile: HPCProfile, state_root: str | Path, *,
                 runner: Callable[[list[str]], Any] | None = None, allow_remote: bool = False):
        self.profile = profile
        self.root = Path(state_root).expanduser().resolve() / profile.profile_id
        self.runner = runner or self._subprocess_runner
        self.allow_remote = allow_remote

    @staticmethod
    def _subprocess_runner(argv: list[str]):
        return subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)

    def _gate(self):
        if not self.allow_remote:
            raise RemoteExecutionDisabled("remote inference execution is disabled; explicit authorization is required")

    def _directory(self, request_id: str) -> Path:
        if not re.fullmatch(r"inference-[0-9a-f]{24}", request_id):
            raise ValueError("invalid inference request id")
        return self.root / request_id

    @contextmanager
    def _lock(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _load(self, directory: Path) -> tuple[dict[str, Any], InferenceJobRequest]:
        manifest = json.loads((directory / "manifest.json").read_text())
        request = InferenceJobRequest.from_dict(json.loads((directory / "request.json").read_text()))
        if manifest["input_hash"] != request.input_hash or manifest["profile_fingerprint"] != self.profile.fingerprint:
            raise IdentityMismatch("stored request or backend profile changed")
        return manifest, request

    def _save(self, directory: Path, manifest: dict[str, Any], action: str, **details):
        now = datetime.now(timezone.utc).isoformat()
        manifest["updated_at"] = now
        manifest.setdefault("events", []).append({"at": now, "action": action, **details})
        _atomic_json(directory / "manifest.json", manifest)

    def _ssh(self, command: str) -> list[str]:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self.profile.controlpath:
            argv += ["-o", "ControlPath=" + self.profile.controlpath]
        return argv + [self.profile.login_host, command]

    def _scp(self, source: str, destination: str) -> list[str]:
        argv = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self.profile.controlpath:
            argv += ["-o", "ControlPath=" + self.profile.controlpath]
        return argv + ["-r", source, destination]

    def _run(self, argv: list[str]):
        self._gate()
        completed = self.runner(argv)
        if completed.returncode:
            raise TransportError(f"transport exit {completed.returncode}: {(completed.stderr or '')[-1000:]}")
        return completed

    def prepare(self, request: InferenceJobRequest, command: Sequence[str], *,
                input_sources: Mapping[str, str | Path] | None = None,
                checkpoint: CheckpointRef | None = None,
                checkpoint_source: str | Path | None = None) -> dict[str, Any]:
        if not command or any(not isinstance(x, str) or not x or "\x00" in x for x in command):
            raise ValueError("trusted worker command must be an explicit argv list")
        directory = self._directory(request.request_id)
        binding = content_hash({"command": list(command), "checkpoint": checkpoint.to_dict() if checkpoint else None})
        if checkpoint:
            checkpoint.validate(request, self.runtime_fingerprint(command))
            if checkpoint_source is None:
                raise ValueError("checkpoint_source is required")
        with self._lock(directory):
            if (directory / "manifest.json").exists():
                manifest, stored = self._load(directory)
                if stored.input_hash != request.input_hash or manifest["command_fingerprint"] != binding:
                    raise IdentityMismatch("existing request has a different command or checkpoint")
                return manifest
            sources = dict(input_sources or {})
            if set(sources) != {x.path for x in request.artifacts}:
                raise ValueError("input_sources must match the declared input artifacts exactly")
            remote = self.profile.remote_root.rstrip("/") + "/inference_jobs/" + request.request_id
            replacements = {"request": remote + "/request.json", "output": remote + "/output",
                            "input": remote, "cpus": str(self.profile.cpus_per_task),
                            "python": self.profile.python_executable,
                            "checkpoint": remote + "/checkpoint/" + checkpoint.artifact.path if checkpoint else None}
            expanded = []
            for argument in command:
                for key, value in replacements.items():
                    token = "{" + key + "}"
                    if token in argument:
                        if value is None:
                            raise ValueError(f"worker command requires unconfigured {key}")
                        argument = argument.replace(token, value)
                expanded.append(argument)
            bundle = directory / "bundle"
            if bundle.exists():
                shutil.rmtree(bundle)
            bundle.mkdir()
            _atomic_json(directory / "request.json", request.to_dict())
            _atomic_json(bundle / "request.json", request.to_dict())
            for artifact in request.artifacts:
                if PurePosixPath(artifact.path).parts[0] in {"request.json", "submit.slurm", "checkpoint.json", "checkpoint", "output"}:
                    raise ValueError("input artifact uses a reserved job path")
                target = bundle / artifact.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(sources[artifact.path], target)
                artifact.validate_file(bundle)
            if checkpoint:
                target = bundle / "checkpoint" / checkpoint.artifact.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint_source, target)
                checkpoint.artifact.validate_file(bundle / "checkpoint")
                _atomic_json(bundle / "checkpoint.json", checkpoint.to_dict())
            # Checkpoint identity validation is not worker resume support.
            # Override any cluster default that would restart this worker.
            directives = ["#!/bin/bash", "#SBATCH --job-name=" + request.request_id,
                          "#SBATCH --partition=" + self.profile.partition, "#SBATCH --ntasks=1",
                          "#SBATCH --cpus-per-task=" + str(self.profile.cpus_per_task),
                          "#SBATCH --time=" + self.profile.walltime, "#SBATCH --no-requeue"]
            if self.profile.account:
                directives.append("#SBATCH --account=" + self.profile.account)
            directives += ["set -euo pipefail", "cd " + shlex.quote(remote), "mkdir -p output",
                           "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                           "export ANDIE_REQUEST_ID=" + shlex.quote(request.request_id),
                           "export ANDIE_INPUT_HASH=" + shlex.quote(request.input_hash),
                           "export ANDIE_RUNTIME_FINGERPRINT=" + shlex.quote(self.runtime_fingerprint(command)),
                           "exec srun " + shlex.join(expanded)]
            (bundle / "submit.slurm").write_text("\n".join(directives) + "\n")
            artifacts = [ArtifactRef.from_file(str(path.relative_to(bundle)), path).to_dict()
                         for path in sorted(bundle.rglob("*")) if path.is_file()]
            manifest = {"schema_version": 1, "request_id": request.request_id, "input_hash": request.input_hash,
                        "profile_id": self.profile.profile_id, "profile_fingerprint": self.profile.fingerprint,
                        "runtime_fingerprint": self.runtime_fingerprint(command), "command_fingerprint": binding,
                        "remote_directory": remote, "bundle_artifacts": artifacts, "job_id": None,
                        "state": "prepared", "submission_attempted": False,
                        "checkpoint": checkpoint.to_dict() if checkpoint else None}
            self._save(directory, manifest, "prepared")
            return manifest

    def runtime_fingerprint(self, command: Sequence[str]) -> str:
        """Checkpoint binding includes profile, runtime version, and trusted argv.

        Worker/code hashes belong in request provenance or input artifacts. This
        records configured identity; it does not attest the remote environment.
        """
        return content_hash({"profile": self.profile.fingerprint, "command": list(command)})

    def read_manifest(self, request_id: str) -> dict[str, Any]:
        directory = self._directory(request_id)
        with self._lock(directory):
            return self._load(directory)[0]

    def submit(self, request_id: str) -> dict[str, Any]:
        self._gate()
        directory = self._directory(request_id)
        with self._lock(directory):
            manifest, _ = self._load(directory)
            if manifest["job_id"]:
                return manifest
            if manifest["submission_attempted"]:
                raise UnknownSubmission("submission outcome is uncertain; reconcile the recorded attempt without retrying")
            bundle = directory / "bundle"
            paths = list(bundle.rglob("*"))
            if any(path.is_symlink() for path in paths):
                raise IdentityMismatch("staged bundle contains a symlink")
            actual = {str(path.relative_to(bundle)) for path in paths if path.is_file()}
            if actual != {x["path"] for x in manifest["bundle_artifacts"]}:
                raise IdentityMismatch("staged bundle file inventory changed")
            for artifact in manifest["bundle_artifacts"]:
                ArtifactRef.from_dict(artifact).validate_file(bundle)
            remote = manifest["remote_directory"]
            manifest["state"] = "staging"
            self._save(directory, manifest, "staging")
            try:
                self._run(self._ssh("mkdir -p " + shlex.quote(remote)))
                self._run(self._scp(str(directory / "bundle") + "/.", self.profile.login_host + ":" + remote + "/"))
            except Exception as error:
                manifest["state"] = "stage_failed"
                self._save(directory, manifest, "stage_failed", error=str(error))
                raise
            # Write-ahead marker: once sbatch could be sent, a crash/timeout or
            # unparsable response must never be interpreted as safe to retry.
            manifest["state"] = "submitting"
            manifest["submission_attempted"] = True
            self._save(directory, manifest, "submission_attempted")
            try:
                response = self._run(self._ssh("cd " + shlex.quote(remote) + " && sbatch --parsable --comment="
                                               + shlex.quote(request_id) + " submit.slurm"))
                match = re.fullmatch(r"(?:Submitted batch job )?([0-9]+)(?:;[A-Za-z0-9_.-]+)?", response.stdout.strip())
                if not match:
                    raise TransportError("sbatch response did not identify exactly one job")
            except Exception as error:
                manifest["state"] = "submission_unknown"
                self._save(directory, manifest, "submission_unknown", error=str(error))
                raise UnknownSubmission("sbatch may have accepted this request; automatic resubmission is blocked") from error
            manifest["job_id"] = match.group(1)
            manifest["state"] = "submitted"
            self._save(directory, manifest, "submitted", job_id=manifest["job_id"])
            return manifest

    @staticmethod
    def _job_id(manifest: Mapping[str, Any]) -> str:
        job_id = manifest.get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9]+", job_id):
            raise UnknownSubmission("no confirmed scheduler job id; reconcile the submission before remote lifecycle operations")
        return job_id

    def status(self, request_id: str) -> dict[str, Any]:
        self._gate()
        directory = self._directory(request_id)
        with self._lock(directory):
            manifest, _ = self._load(directory)
            job_id = self._job_id(manifest)
            try:
                response = self._run(self._ssh("squeue --jobs=" + job_id + " --noheader --format='%i|%T'"))
                source = "squeue"
                rows = [line.split("|") for line in response.stdout.splitlines() if line.strip()]
                rows = [row for row in rows if row[0].strip() == job_id]
                if not rows:
                    # Empty squeue is success for a finished job, not a shell error.
                    source = "sacct"
                    response = self._run(self._ssh("sacct --jobs=" + job_id
                                                   + " --noheader --parsable2 --format=JobIDRaw,State,ExitCode"))
                    rows = [line.split("|") for line in response.stdout.splitlines() if line.strip()]
                    rows = [row for row in rows if row[0].strip() == job_id]
                if len(rows) > 1:
                    raise TransportError("scheduler returned ambiguous parent-job records")
                scheduler_state = rows[0][1].strip().split()[0].rstrip("+") if rows and len(rows[0]) > 1 else "UNKNOWN"
                exit_code = rows[0][2].strip() if rows and len(rows[0]) > 2 else None
                states = {"PENDING": "pending", "CONFIGURING": "pending", "RUNNING": "running",
                          "COMPLETING": "running", "SUSPENDED": "suspended", "REQUEUED": "pending",
                          "RESIZING": "running", "COMPLETED": "scheduler_completed", "CANCELLED": "cancelled",
                          "FAILED": "failed", "TIMEOUT": "failed", "OUT_OF_MEMORY": "failed",
                          "NODE_FAIL": "failed", "PREEMPTED": "preempted", "BOOT_FAIL": "failed",
                          "DEADLINE": "failed", "REVOKED": "failed"}
                state = states.get(scheduler_state, "unknown")
                if scheduler_state == "COMPLETED" and exit_code not in {None, "0:0"}:
                    state = "failed"
                # A validated result is durable even after Slurm accounting expires.
                if manifest["state"] not in {"completed", "result_failed", "unavailable"}:
                    manifest["state"] = state
                manifest["scheduler"] = {"source": source, "state": scheduler_state, "exit_code": exit_code}
                self._save(directory, manifest, "status", scheduler=manifest["scheduler"])
            except Exception as error:
                self._save(directory, manifest, "status_error", error=str(error))
                raise
            return manifest

    def cancel(self, request_id: str) -> dict[str, Any]:
        self._gate()
        directory = self._directory(request_id)
        with self._lock(directory):
            manifest, _ = self._load(directory)
            job_id = self._job_id(manifest)
            if manifest["state"] in {"cancel_requested", "cancelled", "scheduler_completed", "completed", "failed",
                                     "result_failed", "unavailable"}:
                return manifest
            try:
                self._run(self._ssh("scancel " + job_id))
            except Exception as error:
                self._save(directory, manifest, "cancel_error", error=str(error))
                raise
            manifest["state"] = "cancel_requested"
            self._save(directory, manifest, "cancel_requested")
            return manifest

    def fetch(self, request_id: str) -> InferenceJobResult:
        self._gate()
        directory = self._directory(request_id)
        with self._lock(directory):
            manifest, request = self._load(directory)
            job_id = self._job_id(manifest)
            result_directory = directory / "result"
            if result_directory.exists():
                if manifest.get("result_sha256") and file_hash(result_directory / "result.json") != manifest["result_sha256"]:
                    raise IdentityMismatch("cached result envelope changed")
                result = InferenceJobResult.from_dict(json.loads((result_directory / "result.json").read_text()))
                result.validate(request, job_id, result_directory)
                if not manifest.get("result_sha256") and result.status == "completed" and manifest["state"] != "scheduler_completed":
                    raise IdentityMismatch("unregistered successful result lacks scheduler completion")
                if not manifest.get("result_sha256"):
                    # Recover a crash between publishing the validated directory
                    # and recording its manifest, without fetching/resubmitting.
                    manifest["result_sha256"] = file_hash(result_directory / "result.json")
                    manifest["result_path"] = str(result_directory / "result.json")
                    manifest["state"] = {"completed": "completed", "failed": "result_failed", "unavailable": "unavailable"}[result.status]
                    self._save(directory, manifest, "result_recovered")
                return result
            if manifest["state"] not in {"scheduler_completed", "failed", "cancelled", "preempted"}:
                raise InferenceJobError("fetch requires a confirmed terminal scheduler state; poll status first")
            # A new empty destination prevents a previous job/result from being
            # selected by filename. Only this exact result.json is authoritative.
            temporary = Path(tempfile.mkdtemp(prefix=".fetch-", dir=directory))
            try:
                remote = manifest["remote_directory"] + "/output/."
                self._run(self._scp(self.profile.login_host + ":" + remote, str(temporary) + "/"))
                result = InferenceJobResult.from_dict(json.loads((temporary / "result.json").read_text()))
                result.validate(request, job_id, temporary)
                if result.status == "completed" and manifest["state"] != "scheduler_completed":
                    raise IdentityMismatch("successful inference result conflicts with unsuccessful scheduler state")
                result_hash = file_hash(temporary / "result.json")
                if result_directory.exists():
                    raise IdentityMismatch("unregistered result directory exists; refusing to overwrite it")
                os.replace(temporary, result_directory)
                manifest["result_sha256"] = result_hash
                manifest["result_path"] = str(result_directory / "result.json")
                manifest["state"] = {"completed": "completed", "failed": "result_failed", "unavailable": "unavailable"}[result.status]
                self._save(directory, manifest, "result_validated", result_sha256=result_hash)
                return result
            except Exception as error:
                self._save(directory, manifest, "fetch_error", error=str(error))
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
